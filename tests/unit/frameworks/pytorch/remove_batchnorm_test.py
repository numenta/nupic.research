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
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from nupic.research.frameworks.pytorch.model_compare import compare_models
from nupic.research.frameworks.pytorch.models import LeSparseNet
from nupic.research.frameworks.pytorch.remove_batchnorm import remove_batchnorm
from nupic.torch.models.sparse_cnn import gsc_sparse_cnn
from nupic.torch.modules.sparse_weights import rezero_weights

BATCH_NORM_CLASSES = (nn.BatchNorm1d, nn.BatchNorm2d,)


def create_simple_cnn(in_channels=1, cnn_out_channels=2, linear_units=3,
                      sparse_weights=False):
    """
    Simple CNN model for testing batch norm removal. One CNN layer plus one
    fully connected layer plus a linear output layer.  The input shape will be
    (in_channels, 32, 32) and the net will have 12 output classes.
    """
    return LeSparseNet(
        input_shape=(in_channels, 32, 32),
        cnn_out_channels=(cnn_out_channels,),
        cnn_activity_percent_on=(1.0,),
        cnn_weight_percent_on=(1.0,),
        linear_n=(linear_units,),
        linear_activity_percent_on=(1.0,),
        linear_weight_percent_on=(0.5 if sparse_weights else 1.0,),
        boost_strength=0.0,
        boost_strength_factor=1.0,
        use_batch_norm=True,
        dropout=0.0,
        num_classes=12,
        k_inference_factor=1.0
    )


def train_randomly(model, in_channels=1, num_samples=20):
    """
    Train the model on random inputs to ensure the batchnorm really learns something.
    """
    # Create a random training set
    x = torch.randn((num_samples,) + (in_channels, 32, 32))
    targets = torch.randint(0, 12, (num_samples,))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.2)
    model.train()
    for _ in range(10):
        optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, targets)
        loss.backward()
        optimizer.step()


class RemoveBatchnormTest(unittest.TestCase):

    def test_simple_cnn(self):
        """Compare a network with itself after batchnorm is removed."""
        model = create_simple_cnn()
        train_randomly(model)
        model2 = remove_batchnorm(model)

        expected_modules = set(name for name, m in model.named_children()
                               if not isinstance(m, BATCH_NORM_CLASSES))
        actual_modules = set(name for name, m in model2.named_children())

        self.assertEqual(actual_modules, expected_modules)
        self.assertTrue(compare_models(model, model2, (1, 32, 32)))

    def test_cnn_more_out_channels(self):
        """Compare another network with itself after batchnorm is removed."""
        model = create_simple_cnn(
            cnn_out_channels=16,
            linear_units=20,
        )
        train_randomly(model)
        model2 = remove_batchnorm(model)
        expected_modules = set(name for name, m in model.named_children()
                               if not isinstance(m, BATCH_NORM_CLASSES))
        actual_modules = set(name for name, m in model2.named_children())

        self.assertEqual(actual_modules, expected_modules)
        self.assertTrue(compare_models(model, model2, (1, 32, 32)))

    def test_cnn_more_in_channels(self):
        """
        Compare a network with 3 in_channels with itself after batchnorm is removed.
        """
        model = create_simple_cnn(
            in_channels=3,
            cnn_out_channels=4,
            linear_units=5,
        )
        train_randomly(model, in_channels=3)
        model2 = remove_batchnorm(model)

        expected_modules = set(name for name, m in model.named_children()
                               if not isinstance(m, BATCH_NORM_CLASSES))
        actual_modules = set(name for name, m in model2.named_children())

        self.assertEqual(actual_modules, expected_modules)
        self.assertTrue(compare_models(model, model2, (3, 32, 32)))

    def test_cnn_sparse_weights(self):
        """
        Compare a network with 3 in_channels with itself after batchnorm is removed.
        """
        model = create_simple_cnn(
            in_channels=3,
            cnn_out_channels=4,
            linear_units=5,
            sparse_weights=True,
        )
        train_randomly(model, in_channels=3)
        model.apply(rezero_weights)
        model2 = remove_batchnorm(model)

        expected_modules = set(name for name, m in model.named_children()
                               if not isinstance(m, BATCH_NORM_CLASSES))
        actual_modules = set(name for name, m in model2.named_children())

        self.assertEqual(actual_modules, expected_modules)
        self.assertTrue(compare_models(model, model2, (3, 32, 32)))

    def test_gsc(self):
        """
        Compare the GSC network after batchnorm is removed.
        """
        model = gsc_sparse_cnn(pretrained=True)
        model2 = remove_batchnorm(model)

        expected_modules = set(name for name, m in model.named_children()
                               if not isinstance(m, BATCH_NORM_CLASSES))
        actual_modules = set(name for name, m in model2.named_children())

        self.assertEqual(actual_modules, expected_modules)
        self.assertTrue(compare_models(model, model2, (1, 32, 32)))


if __name__ == "__main__":
    unittest.main()
