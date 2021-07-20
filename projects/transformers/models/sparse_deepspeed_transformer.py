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

        # Sparsify fused attention layer
        device = module.attn_qkvw.device
        out_features = module.attn_qkvw.shape[0] // 3
        in_features = module.attn_qkvw.shape[1]
        mask_qw = _compute_zero_mask(out_features, in_features, self.sparsity, device)
        mask_kw = _compute_zero_mask(out_features, in_features, self.sparsity, device)
        mask_vw = _compute_zero_mask(out_features, in_features, self.sparsity, device)
        zero_mask_attn_qkv = torch.cat((mask_qw, mask_kw, mask_vw), dim=0)
        self.register_buffer("zero_mask_attn_qkv", zero_mask_attn_qkv)

        # Sparsify attention output layer
        zero_mask_attn_out = _compute_zero_mask(
            *module.attn_ow.shape, sparsity=self.sparsity, device=device)
        self.register_buffer("zero_mask_attn_out", zero_mask_attn_out)

        # Sparsify intermediate layer
        zero_mask_inter = _compute_zero_mask(
            *module.inter_w.shape, sparsity=self.sparsity, device=device)
        self.register_buffer("zero_mask_inter", zero_mask_inter)

        # Sparsify output layer
        zero_mask_output = _compute_zero_mask(
            *module.output_w.shape, sparsity=self.sparsity, device=device)
        self.register_buffer("zero_mask_output", zero_mask_output)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)

    def init_transformer_weights(self, *args, **kwargs):
        return self.module.init_transformer_weights(*args, **kwargs)

    def rezero_weights(self):
        self.module.attn_qkvw.data[self.zero_mask_attn_qkv] = 0
        self.module.attn_ow.data[self.zero_mask_attn_out] = 0
        self.module.inter_w.data[self.zero_mask_inter] = 0
        self.module.output_w.data[self.zero_mask_output] = 0
