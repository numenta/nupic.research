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

"""
This __init__ updates the transformer model and config mappings to include custom sparse
Bert models. This way, they may be automatically loaded via AutoModelForMaskedLM and
related utilities.
"""

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    TOKENIZER_MAPPING,
    BertTokenizer,
    BertTokenizerFast,
)

from .configuration_sparse_bert import SparseBertConfig
from .modeling_sparse_bert import (
    SparseBertForMaskedLM,
    SparseBertForSequenceClassification,
)

CONFIG_MAPPING.update(
    sparse_bert=SparseBertConfig
)


TOKENIZER_MAPPING.update({
    SparseBertConfig: (BertTokenizer, BertTokenizerFast),
})


MODEL_FOR_MASKED_LM_MAPPING.update({
    SparseBertConfig: SparseBertForMaskedLM,
})


MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.update({
    SparseBertConfig: SparseBertForSequenceClassification
})