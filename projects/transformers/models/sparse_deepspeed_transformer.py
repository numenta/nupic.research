# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import torch
from deepspeed import DeepSpeedTransformerLayer

from nupic.torch.modules.sparse_weights import SparseWeightsBase

__all__ = ["SparseDeepSpeedTransformerLayer"]


def _compute_zero_mask(out_features, in_features, sparsity, device):
    nnz = int(round((1 - sparsity) * in_features))
    zero_mask = torch.ones(out_features, in_features, dtype=torch.bool, device=device)
    for i in range(out_features):
        nnz_indices = torch.randperm(in_features, device=device)[:nnz]
        zero_mask[i, nnz_indices] = False
    return zero_mask


class SparseDeepSpeedTransformerLayer(SparseWeightsBase):
    """
    This class wraps DeepSpeedTransformerLayer to sparsify the QKV fused weights
    """
    def __init__(self, module, sparsity=None):
        assert isinstance(module, DeepSpeedTransformerLayer)
        super().__init__(module=module, sparsity=sparsity)

        # Sparsify fused attention layer, applying local sparsity to each QKV layer
        device = module.attn_qkvw.device
        out_features = module.attn_qkvw.shape[0] // 3
        in_features = module.attn_qkvw.shape[1]
        mask_qw = _compute_zero_mask(out_features, in_features, self.sparsity, device)
        mask_kw = _compute_zero_mask(out_features, in_features, self.sparsity, device)
        mask_vw = _compute_zero_mask(out_features, in_features, self.sparsity, device)
        zero_mask_attn_qkvw = torch.cat((mask_qw, mask_kw, mask_vw), dim=0)

        # Sparsify attention output layer
        zero_mask_attn_ow = _compute_zero_mask(
            *module.attn_ow.shape, sparsity=self.sparsity, device=device)

        # Sparsify intermediate layer
        zero_mask_inter_w = _compute_zero_mask(
            *module.inter_w.shape, sparsity=self.sparsity, device=device)

        # Sparsify output layer
        zero_mask_output_w = _compute_zero_mask(
            *module.output_w.shape, sparsity=self.sparsity, device=device)

        # Compute contiguous weight tensor and mask
        weights = [module.attn_qkvw, module.attn_ow, module.inter_w, module.output_w]
        flat_weights = torch.cat([w.flatten() for w in weights], dim=0).contiguous()
        start = end = 0
        for w in weights:
            end += w.numel()
            w.data = flat_weights[start:end].view(w.shape)
            start = end

        zero_mask = torch.cat((zero_mask_attn_qkvw.flatten(),
                               zero_mask_attn_ow.flatten(),
                               zero_mask_inter_w.flatten(),
                               zero_mask_output_w.flatten())).contiguous()
        self.register_buffer("zero_mask", zero_mask)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def init_transformer_weights(self, *args, **kwargs):
        return self.module.init_transformer_weights(*args, **kwargs)

    def rezero_weights(self):
        # Unflatten zero_mask
        weights = [self.module.attn_qkvw, self.module.attn_ow,
                   self.module.inter_w, self.module.output_w]

        start = end = 0
        for w in weights:
            end += w.numel()
            w.data[self.zero_mask[start:end].view(w.shape)] = 0
            start = end

    @property
    def weight(self):
        # Return flattened weights
        return torch.cat((self.module.attn_qkvw.data.flatten(),
                          self.module.attn_ow.data.flatten(),
                          self.module.inter_w.data.flatten(),
                          self.module.output_w.data.flatten()))

    @property
    def bias(self):
        return self.module.bias
