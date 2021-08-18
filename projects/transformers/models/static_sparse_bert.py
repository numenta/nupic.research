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

from dataclasses import dataclass
from typing import Union

import torch
from transformers import BertModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertPooler,
)
from transformers.utils import logging

from nupic.research.frameworks.pytorch.model_utils import (
    filter_modules,
    set_module_attr,
)
from nupic.torch.modules import SparseWeights

from .register_bert_model import register_bert_model
from .sparse_embedding import SparseEmbeddings

logger = logging.get_logger(__name__)


__all__ = [
    "StaticSparseEncoderBertModel",
    "StaticSparseNonAttentionBertModel",
    "FullyStaticSparseBertModel",
]


@register_bert_model
class StaticSparseEncoderBertModel(BertModel):
    """
    Sparse Version of Bert that applies static sparsity to all linear layers (included
    attention) in the encoder network. Use by this model declaring
    model_type="static_sparse_encoder_bert".
    """

    @dataclass
    class ConfigKWargs:
        """Keyword arguments to configure sparsity."""
        sparsity: float = 0.9

    def __init__(self, config, add_pooling_layer=True):
        # Call the init one parent class up. Otherwise, the model will be defined twice.
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Sparsify linear modules.
        self.sparsify_model()

        self.init_weights()

    def sparsify_model(self):
        """
        Sparsify all linear layers in encoder.
        """

        encoder = self.encoder
        sparsity = self.config.sparsity
        device = self.device

        # Perform model surgery by replacing the linear layers with `SparseWeights`.
        linear_modules = filter_modules(encoder, include_modules=[torch.nn.Linear])
        for name, module in linear_modules.items():
            sparse_module = SparseWeights(module, sparsity=sparsity).to(device)
            set_module_attr(self.encoder, name, sparse_module)


@register_bert_model
class StaticSparseNonAttentionBertModel(BertModel):
    """
    Sparse Version of Bert that applies static sparsity to all non-attention linear
    layers in the encoder network. Use by this model declaring
    model_type="static_sparse_non_attention_bert".
    """

    @dataclass
    class ConfigKWargs:
        """Keyword arguments to configure sparsity."""
        sparsity: float = 0.5
        num_sparse_layers: int = 12

    def __init__(self, config, add_pooling_layer=True):
        # Call the init one parent class up. Otherwise, the model will be defined twice.
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Sparsify linear modules.
        self.sparsify_model()

        self.init_weights()

    def sparsify_model(self):
        """
        Sparsify all non-attention linear layers in encoder.
        """

        encoder = self.encoder
        num_sparse_layers = self.config.num_sparse_layers
        sparsity = self.config.sparsity
        device = self.device

        for idx in range(num_sparse_layers):
            intermediate_layer = encoder.layer[idx].intermediate.dense
            encoder.layer[idx].intermediate.dense = \
                SparseWeights(intermediate_layer, sparsity=sparsity).to(device)

            output_layer = encoder.layer[idx].output.dense
            encoder.layer[idx].output.dense = \
                SparseWeights(output_layer, sparsity=sparsity).to(device)


@register_bert_model
class FullyStaticSparseBertModel(BertModel):
    """
    Sparse Version of Bert that applies static sparsity to linear layers; every linear
    layer in the encoder as well as the word embedding layer. Use by this model
    declaring model_type="fully_static_sparse_bert".
    """

    @dataclass
    class ConfigKWargs:
        """Keyword arguments to configure sparsity."""
        sparsity: Union[float, dict] = 0.5
        sparsify_all_embeddings: bool = False

    def __init__(self, config, add_pooling_layer=True):
        # Call the init one parent class up. Otherwise, the model will be defined twice.
        BertPreTrainedModel.__init__(self, config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Sparsify linear modules.
        self.sparsify_model()

        self.init_weights()

    def sparsify_model(self):
        """
        Sparsify all linear layers in encoder as well as the word embedding layer.
        """

        encoder = self.encoder
        sparsity = self.config.sparsity
        device = self.device

        # Use `getattr` here for backwards compatibility for configs without this param.
        sparsify_all_embeddings = getattr(self.config, "sparsify_all_embeddings", False)

        def get_sparsity(name):
            if isinstance(sparsity, dict):
                if name in sparsity:
                    return sparsity[name]
                else:
                    raise KeyError(f"Layer {name} not included in sparsity dict.")
            else:
                return sparsity

        # Perform model surgery by replacing the linear layers with `SparseWeights`.
        linear_modules = filter_modules(encoder, include_modules=[torch.nn.Linear])
        for name, module in linear_modules.items():
            layer_sparsity = get_sparsity("bert.encoder." + name)
            sparse_module = SparseWeights(
                module,
                sparsity=layer_sparsity,
                allow_extremes=True  # this allows the model to start fully dense
            )
            set_module_attr(self.encoder, name, sparse_module.to(device))

        # Replace the embedding layers in a similar fashion.
        if sparsify_all_embeddings:
            embeddings = ["word_embeddings",
                          "position_embeddings",
                          "token_type_embeddings"]
        else:
            embeddings = ["word_embeddings"]

        for embedding_name in embeddings:
            dense_module = getattr(self.embeddings, embedding_name)
            layer_sparsity = get_sparsity(f"bert.embeddings.{embedding_name}")
            sparse_module = SparseEmbeddings(dense_module, sparsity=layer_sparsity)
            setattr(self.embeddings, embedding_name, sparse_module.to(device))

# Import new class to override embedding related functions. # noqa
# This class was implicitly defined through register_bert_model # noqa
from . import FullyStaticSparseBertForMaskedLM  # noqa


def _get_resized_sparse_embeddings(self, old_embeddings, new_num_tokens=None):
    """
    Build a resized Embedding Module from a provided token Embedding Module. Increasing
    the size will add newly initialized vectors at the end. Reducing the size will
    remove vectors from the end.

    This is modified from the `Transformers`_ repo to work with SparseEmbeddings modules

    .. _Transformers:
        https://github.com/huggingface/transformers/blob/0d909f6bd8ca0bc1ec8f42e089b64b4fffc4d230/src/transformers/modeling_utils.py#L645

    :param old_embeddings: Old embeddings to be resized.
    :type old_embeddings: SparseEmbeddings module
    :param new_num_tokens: New number of tokens in the embedding matrix.
    :type new_num_tokens: int
    """
    if new_num_tokens is None:
        return old_embeddings

    old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
    if old_num_tokens == new_num_tokens:
        return old_embeddings

    if not isinstance(old_embeddings, SparseEmbeddings):
        raise TypeError(
            f"Old embeddings are of type {type(old_embeddings)}, which is not an "
            f"instance of {SparseEmbeddings}. You should either use a different"
            "resize function or make sure that `old_embeddings` are an instance "
            f"of {SparseEmbeddings}."
        )

    # Build new embeddings
    new_embeddings = torch.nn.Embedding(new_num_tokens, old_embedding_dim)
    new_embeddings = SparseEmbeddings(new_embeddings, sparsity=old_embeddings.sparsity)
    new_embeddings = new_embeddings.to(self.device)

    # initialize all new embeddings (in particular added tokens)
    self._init_weights(new_embeddings)

    # Copy token embeddings from the previous weights
    num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
    old_data = old_embeddings.weight.data[:num_tokens_to_copy, :]
    new_embeddings.weight.data[:num_tokens_to_copy, :] = old_data

    return new_embeddings


FullyStaticSparseBertForMaskedLM._get_resized_embeddings = (
    _get_resized_sparse_embeddings
)
