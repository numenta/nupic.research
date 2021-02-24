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
from torch import nn
from transformers import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertPreTrainedModel,
    BertModel,
)
from transformers.models.bert.modeling_bert import (
    BertEmbeddings,
    BertEncoder,
    BertOnlyMLMHead,
    BertPooler,
)
from transformers.utils import logging

from nupic.research.frameworks.pytorch.model_utils import (
    filter_modules,
    set_module_attr,
)
from nupic.torch.modules import SparseWeights

logger = logging.get_logger(__name__)


class SparseBertModel(BertModel):
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


class SparseBertForMaskedLM(BertForMaskedLM):
    """
    Bert Model with a `language modeling` head on top.

    Calls SparseBert in forward.
    """
    def __init__(self, config):

        # Call the init one parent class up. Otherwise, the model will be defined twice.
        BertPreTrainedModel.__init__(self, config)

        if config.is_decoder:
            logger.warning(
                # This warning was included with the original BertForMaskedLM.
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = SparseBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()


class SparseBertForSequenceClassification(BertForSequenceClassification):
    """
    Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
    output) e.g. for GLUE tasks.

    Calls SparseBert in forward.
    """,
    def __init__(self, config):
        BertPreTrainedModel.__init__(self, config)
        self.num_labels = config.num_labels

        # Replace `BertModel` with SparseBertModel.
        self.bert = SparseBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()
