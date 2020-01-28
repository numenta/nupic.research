#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
import io
import unittest

import torch
import torch.nn

from nupic.research.frameworks.pytorch.model_compare import compare_models
from nupic.research.frameworks.pytorch.model_utils import (
    count_nonzero_params,
    deserialize_state_dict,
    serialize_state_dict,
)
from nupic.research.frameworks.pytorch.models.le_sparse_net import LeSparseNet
from nupic.torch.modules import Flatten


def simple_linear_net():
    return torch.nn.Sequential(
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 2)
    )


def simple_conv_net():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 3, 5),
        torch.nn.MaxPool2d(2),
        torch.nn.ReLU(),
        Flatten(),
        torch.nn.Linear(111, 3),
        torch.nn.ReLU(),
        torch.nn.Linear(3, 2)
    )


def simple_sparse_conv_net():
    return LeSparseNet(
        input_shape=(1, 8, 8),
        cnn_out_channels=(4, 6),
        cnn_weight_percent_on=(0.2, 0.5),
        linear_n=(100,),
        linear_activity_percent_on=(0.1,),
        linear_weight_percent_on=(0.2,),
        use_batch_norm=True,
        num_classes=2,
    )


class ModelCompareTest(unittest.TestCase):

    def test_simple_linear(self):
        """Count non-zero params in a simple linear net"""

        model = simple_linear_net()
        expected_params = 32 * 16 + 2 * 16 + 16 + 2
        total_params, total_nonzero_params = count_nonzero_params(model)
        self.assertEqual(total_nonzero_params, expected_params)
        self.assertEqual(total_params, expected_params)

        model[0].weight[0, 0] = 0.0
        model[0].weight[0, 1] = 0.0
        model[2].weight[0, 0] = 0.0
        model[2].weight[1, 0] = 0.0

        total_params, total_nonzero_params = count_nonzero_params(model)
        self.assertEqual(total_nonzero_params, expected_params - 4)
        self.assertEqual(total_params, expected_params)

    def test_simple_conv_net(self):
        """Count non-zero params in a simple linear net"""

        model = simple_conv_net()
        expected_params = 75 + 3 + 3 * 111 + 3 + 6 + 2
        total_params, total_nonzero_params = count_nonzero_params(model)
        self.assertEqual(total_nonzero_params, expected_params)
        self.assertEqual(total_params, expected_params)

        model[0].weight[0, 0, 3, 3] = 0.0
        model[0].weight[1, 0, 1, 1] = 0.0
        model[4].weight[0, 0] = 0.0
        model[4].weight[1, 0] = 0.0

        total_params, total_nonzero_params = count_nonzero_params(model)
        self.assertEqual(total_nonzero_params, expected_params - 4)
        self.assertEqual(total_params, expected_params)

    def test_simple_sparse_conv_net(self):
        """Count non-zero params in a simple linear net"""

        model = simple_sparse_conv_net()
        expected_params = (4 * 5 * 5 + 4) + (4 * 5 * 5 * 6 + 6) + \
                          (6 * 100 + 100) + (100 * 2 + 2)
        expected_nonzero_params = (4 * round(0.2 * 5 * 5) + 4) + \
                                  (6 * round(0.5 * 5 * 5 * 4) + 6) + \
                                  (100 * round(0.2 * 6) + 100) + \
                                  (100 * 2 + 2)
        total_params, total_nonzero_params = count_nonzero_params(model)
        self.assertEqual(total_nonzero_params, expected_nonzero_params)
        self.assertEqual(total_params, expected_params)


class ModelSerializationTest(unittest.TestCase):

    def test_serialization(self):
        model1 = simple_linear_net()
        model2 = simple_linear_net()

        def init(m):
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.data.fill_(42.0)
        model2.apply(init)

        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, model1.state_dict())

            buffer.seek(0)
            state_dict = deserialize_state_dict(buffer)
            model2.load_state_dict(state_dict)

        self.assertTrue(compare_models(model1, model2, (32,)))


if __name__ == "__main__":
    unittest.main()
