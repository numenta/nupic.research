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

import tempfile
import unittest

from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
)

from nupic.torch.modules import SparseWeights
# FIXME: Importing module relative to the project
from projects.transformers.models import (
    SparseEmbeddings,
    StaticSparseEncoderBertConfig,
    StaticSparseEncoderBertForMaskedLM,
    StaticSparseEncoderBertForSequenceClassification,
)


class SparseBertModelTest(unittest.TestCase):
    """
    Test the ability to load and save StaticSparseEncoderBert models and associated
    StaticSparseEncoderBertConfig.
    """

    def setUp(self):

        # Config for model with sparse encoder.
        self.config = CONFIG_MAPPING["static_sparse_encoder_bert"](
            sparsity=0.90,
            num_attention_heads=1,
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=64 * 4,
        )

        # Config for model with sparse encoder and sparse embedding layer.
        self.fully_sparse_config = CONFIG_MAPPING["fully_static_sparse_bert"](
            sparsity=0.75,
            num_attention_heads=1,
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=64 * 4,
        )

    def test_model_mappings(self):
        config_cls = StaticSparseEncoderBertConfig
        self.assertTrue(config_cls in MODEL_FOR_MASKED_LM_MAPPING)
        self.assertTrue(config_cls in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING)

    def test_sparse_bert_config(self):
        self.assertIsInstance(self.config, StaticSparseEncoderBertConfig)
        self.assertEqual(self.config.sparsity, 0.9)

    def test_auto_model_for_mask_lm(self):

        model = AutoModelForMaskedLM.from_config(self.config)
        self.assertIsInstance(model, StaticSparseEncoderBertForMaskedLM)

        # The are two layers with six linear layers and only one head.
        # There should be a total of 12 linear layers.
        sparse_layers = []
        for module in model.modules():
            if isinstance(module, SparseWeights):
                sparse_layers.append(module)
        self.assertEqual(12, len(sparse_layers))

    def test_auto_model_for_sequence_classification(self):

        model_seq_cls = AutoModelForSequenceClassification.from_config(self.config)
        self.assertIsInstance(
            model_seq_cls,
            StaticSparseEncoderBertForSequenceClassification
        )

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

        self.assertIsInstance(config, StaticSparseEncoderBertConfig)
        self.assertIsInstance(model, StaticSparseEncoderBertForMaskedLM)
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

    def test_model_with_sparse_embedding(self):
        # Init model type of fully_static_sparse_bert.
        model = AutoModelForMaskedLM.from_config(self.fully_sparse_config)

        # Call resize on token embeddings. This required overriding
        # _get_resized_sparse_embeddings to work with a SparseEmbeddings layer.
        model.resize_token_embeddings(30000)

        # Validate the word embeddings are sparse.
        word_embeddings = model.bert.embeddings.word_embeddings
        self.assertIsInstance(word_embeddings, SparseEmbeddings)
        word_embeddings.rezero_weights()

        num_zero = (word_embeddings.weight == 0).sum()
        self.assertTrue(num_zero >= 1440000)

        # Make the on weights all one in case there are random zeros.
        word_embeddings.weight.data[:] = 1
        word_embeddings.rezero_weights()

        # Validate sparsity per embedding. Should be 64 * 0.75
        num_embeddings = word_embeddings.module.num_embeddings
        for n in range(num_embeddings):
            num_zero = (word_embeddings.weight[n, :] == 0).sum()
            self.assertEqual(num_zero, 48)


if __name__ == "__main__":
    unittest.main(verbosity=2)
