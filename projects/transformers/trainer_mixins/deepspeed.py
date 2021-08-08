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
from collections import defaultdict

import torch
from deepspeed import replace_transformer_layer
from deepspeed.module_inject import HFBertLayerPolicy
from transformers import BertLayer

from nupic.torch.modules import SparseWeights
from projects.transformers.models.sparse_deepspeed_transformer import (
    SparseDeepSpeedTransformerLayer,
)

__all__ = ["DeepspeedTransformerLayerMixin", "replace_sparse_transformer_layer"]


def replace_sparse_transformer_layer(model, config=None, fp16=True, training=True):

    # Store sparse mask and properties before deepspeed conversion
    sparsity_per_layer = defaultdict(dict)
    for i, bert_layer in enumerate(model.encoder.layer):
        # Only supports huggingface BertLayer
        assert isinstance(bert_layer, BertLayer)

        # Check if model has sparse bert layers
        query = bert_layer.attention.self.query
        if isinstance(query, SparseWeights):
            key = bert_layer.attention.self.key
            value = bert_layer.attention.self.value
            intermediate = bert_layer.intermediate.dense
            attn_out = bert_layer.attention.output.dense
            output = bert_layer.output.dense

            # Make sure all layers are sparse
            assert isinstance(key, SparseWeights)
            assert isinstance(value, SparseWeights)
            assert isinstance(intermediate, SparseWeights)
            assert isinstance(attn_out, SparseWeights)
            assert isinstance(output, SparseWeights)

            # Assume same sparsity for all layers
            sparsity = query.sparsity
            assert key.sparsity == sparsity
            assert value.sparsity == sparsity
            assert intermediate.sparsity == sparsity
            assert attn_out.sparsity == sparsity
            assert output.sparsity == sparsity

            # Create flatten zero mask
            zero_mask = torch.cat(
                (query.zero_mask.flatten(),
                 key.zero_mask.flatten(),
                 value.zero_mask.flatten(),
                 attn_out.zero_mask.flatten(),
                 intermediate.zero_mask.flatten(),
                 output.zero_mask.flatten(),
                 )
            )
            sparsity_per_layer[i] = {
                "sparsity": sparsity,
                "zero_mask": zero_mask,
            }

    # Convert transform layers to deepspeed.
    ds_model = replace_transformer_layer(
        BertLayer,
        model,
        policy=HFBertLayerPolicy,
        fp16=fp16,
        config=config,
        training=training,
        encoder_decoder=True,
    )

    # This conversion replaces all HF BertLayer with DeepSpeedTransformerLayer.
    # In consequence any SparseWeight layer manually injected into the original
    # BertLayer will be replaced with a regular nn.Linear layer. For this reason
    # we need to convert all DeepSpeedTransformerLayer to
    # SparseDeepSpeedTransformerLayer and restore the original zero mask and
    # sparsity configuration
    for i, sparse_config in sparsity_per_layer.items():
        ds_layer = ds_model.encoder.layer[i]

        sparsity = sparse_config["sparsity"]
        sparse_layer = SparseDeepSpeedTransformerLayer(ds_layer, sparsity=sparsity)

        # Update masks based on original sparse layer
        zero_mask = sparse_config["zero_mask"]
        sparse_layer.zero_mask = zero_mask.bool()
        sparse_layer.rezero_weights()

        ds_model.encoder.layer[i] = sparse_layer

    return ds_model


class DeepspeedTransformerLayerMixin:
    """
    Mixin to HF Trainer class used to replace HF transform layers with deepspeed
    fused QKV transform layer.

    .. note::
        Assume same sparsity for all layers

    See https://www.deepspeed.ai/news/2020/05/27/fastest-bert-training.html#bert-highly-optimized-transformer-kernels
    """  # noqa: E501

    def __init__(self, model, args, **kwargs):
        super().__init__(model=model, args=args, **kwargs)
        # Make sure deepspeed is enabled
        assert args.deepspeed is not None
        base_model = getattr(model, "base_model", model)
        config = getattr(model, "config", None)
        replace_sparse_transformer_layer(
            model=base_model, config=config, fp16=args.fp16
        )
