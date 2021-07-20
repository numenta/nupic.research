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
import copy
import sys
import unittest
from os.path import expanduser

import torch
from transformers import CONFIG_MAPPING, AutoModelForMaskedLM

from nupic.research.frameworks.pytorch.model_utils import set_random_seed
from nupic.torch.modules import rezero_weights

sys.path.insert(0, expanduser("~/nta/nupic.research/projects/transformers"))  # noqa
import models  # noqa :F401
from trainer_mixins.deepspeed import replace_sparse_transformer_layer  # noqa


class SparseBertModelTest(unittest.TestCase):
    def setUp(self):
        set_random_seed(42)
        self.device = torch.device("cuda")

        # Config for model with sparse encoder and sparse embedding layer.
        self.config = CONFIG_MAPPING["fully_static_sparse_bert"](
            num_attention_heads=2,
            num_hidden_layers=2,
            hidden_size=128,
            intermediate_size=512,
            max_position_embeddings=128,
            sparsity=0.75,
        )
        self.sparse_model = AutoModelForMaskedLM.from_config(self.config)
        self.sparse_model.resize_token_embeddings()
        self.sparse_model.apply(rezero_weights)

    def test_sparse_replace_transformer_layer(self):
        original_model = copy.deepcopy(self.sparse_model).half()
        deepspeed_model = copy.deepcopy(original_model)
        deepspeed_model.to(self.device)
        original_model.to(self.device)

        replace_sparse_transformer_layer(
            model=deepspeed_model.base_model,
            config=self.config,
            training=True,
            fp16=True,
        )

        training_steps = 5
        batch_size = 10
        num_embeddings = self.config.max_position_embeddings
        attention_mask = torch.ones(
            training_steps,
            batch_size,
            num_embeddings,
            dtype=torch.half,
            device=self.device,
        )
        input_ids = torch.ones(
            training_steps,
            batch_size,
            num_embeddings,
            dtype=torch.long,
            device=self.device,
        )
        token_type_ids = torch.ones(
            training_steps,
            batch_size,
            num_embeddings,
            dtype=torch.long,
            device=self.device,
        )
        labels = torch.ones(
            training_steps,
            batch_size,
            num_embeddings,
            dtype=torch.long,
            device=self.device,
        )

        # Train for a few steps
        for i in range(training_steps):
            actual_outputs = deepspeed_model(
                attention_mask=attention_mask[i],
                input_ids=input_ids[i],
                labels=labels[i],
                token_type_ids=token_type_ids[i],
            )
            actual_loss = actual_outputs.loss

            expected_outputs = original_model(
                attention_mask=attention_mask[i],
                input_ids=input_ids[i],
                labels=labels[i],
                token_type_ids=token_type_ids[i],
            )
            expected_loss = expected_outputs.loss

            actual_loss.backward()
            expected_loss.backward()

            # Make sure model returns same results
            self.assertTrue(torch.isclose(actual_loss, expected_loss, atol=1e-2))

            deepspeed_model.apply(rezero_weights)
            original_model.apply(rezero_weights)
