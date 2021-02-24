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

logger = logging.get_logger(__name__)


@register_bert_model
class SparseBertModel(BertModel):
    """
    Sparse Version of Bert that applies static sparsity to all linear layers (included
    attention) in the encoder network.
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
