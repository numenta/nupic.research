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

import sys
import tempfile
import unittest
from os.path import expanduser

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

sys.path.insert(0, expanduser("~/nta/nupic.research/projects/transformers")) # noqa
from models import (
    SparseBertConfig,
    SparseBertForMaskedLM,
    SparseBertForSequenceClassification,
)
from nupic.torch.modules import SparseWeights
# noqa


class SparseBertModelTest(unittest.TestCase):
    """
    Test the ability to load and save SparseBert models and associated SparseBertConfig.
    """

    def setUp(self):
        self.config = CONFIG_MAPPING["sparse_bert"](
            sparsity=0.9,
            num_attention_heads=1,
            num_hidden_layers=2,
        )

    def test_model_mappings(self):
        self.assertTrue(SparseBertConfig in MODEL_FOR_MASKED_LM_MAPPING)
        self.assertTrue(SparseBertConfig in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING)

    def test_sparse_bert_config(self):
        self.assertIsInstance(self.config, SparseBertConfig)
        self.assertEqual(self.config.sparsity, 0.9)

    def test_auto_model_for_mask_lm(self):

        model = AutoModelForMaskedLM.from_config(self.config)
        self.assertIsInstance(model, SparseBertForMaskedLM)

        # The are two layers with six linear layers and only one head.
        # There should be a total of 12 linear layers.
        sparse_layers = []
        for module in model.modules():
            if isinstance(module, SparseWeights):
                sparse_layers.append(module)
        self.assertEqual(12, len(sparse_layers))

    def test_auto_model_for_sequence_classification(self):

        model_seq_cls = AutoModelForSequenceClassification.from_config(self.config)
        self.assertIsInstance(model_seq_cls, SparseBertForSequenceClassification)

        # The are two layers with six linear layers and only one head.
        # There should be a total of 12 linear layers.
        sparse_layers = []
        for module in model_seq_cls.modules():
            if isinstance(module, SparseWeights):
                sparse_layers.append(module)
        self.assertEqual(12, len(sparse_layers))

    def test_save_and_load_from_pretrained(self):

        # Load new model.
        model = AutoModelForMaskedLM.from_config(self.config)

        # Create temp directory to save and load model.
        tempdir = tempfile.TemporaryDirectory()
        model_path = tempdir.name

        # Save model.
        model.save_pretrained(model_path)

        # Load saved model and config.
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        config = model.config

        self.assertIsInstance(config, SparseBertConfig)
        self.assertIsInstance(model, SparseBertForMaskedLM)
        self.assertEqual(config.sparsity, 0.9)

        # The are two layers with six linear layers and only one head.
        # There should be a total of 12 linear layers.
        sparse_layers = []
        for module in model.modules():
            if isinstance(module, SparseWeights):
                sparse_layers.append(module)
        self.assertEqual(12, len(sparse_layers))

        # Cleanup saved model and config.
        tempdir.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
