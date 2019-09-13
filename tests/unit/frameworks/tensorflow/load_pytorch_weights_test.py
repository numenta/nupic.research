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

import numpy as np
import tensorflow as tf
import torch
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.platform import test

from nupic.research.frameworks.tensorflow.utils import load_pytorch_weights
from nupic.tensorflow.models import GSCSparseCNN, GSCSuperSparseCNN


@keras_parameterized.run_all_keras_modes
class LoadPytorchWeightsTest(keras_parameterized.TestCase):

    def setUp(self):
        super(LoadPytorchWeightsTest, self).setUp()
        self.data = np.random.rand(10, 1, 32, 32)

    def test_gsc_sparse_cnn(self):
        """
        Test loading gsc_sparse_cnn from pytorch
        """
        model_pt = torch.hub.load(github="numenta/nupic.torch",
                                  model="gsc_sparse_cnn",
                                  pretrained=True)
        model_pt.eval()

        model_tf = GSCSparseCNN(pre_trained=False)
        load_pytorch_weights(model_tf, model_pt)
        model_tf.compile(optimizer="sgd",
                         loss="sparse_categorical_crossentropy",
                         metrics=["accuracy"])
        data_pt = torch.tensor(self.data, dtype=torch.float32)
        data_tf = tf.convert_to_tensor(self.data.transpose(0, 2, 3, 1),
                                       dtype=tf.float32)

        out_pt = model_pt(data_pt)
        out_tf = model_tf(data_tf, training=False)
        out_tf = tf.log(out_tf)

        self.assertAllClose(out_pt.detach().numpy(), out_tf)

    def test_gsc_super_sparse_cnn(self):
        """
        Test loading gsc_sparse_cnn from pytorch
        """
        model_pt = torch.hub.load(github="numenta/nupic.torch",
                                  model="gsc_super_sparse_cnn",
                                  pretrained=True)
        model_pt.eval()

        model_tf = GSCSuperSparseCNN(pre_trained=False)
        load_pytorch_weights(model_tf, model_pt)
        model_tf.compile(optimizer="sgd",
                         loss="sparse_categorical_crossentropy",
                         metrics=["accuracy"])
        data_pt = torch.tensor(self.data, dtype=torch.float32)
        data_tf = tf.convert_to_tensor(self.data.transpose(0, 2, 3, 1),
                                       dtype=tf.float32)

        out_pt = model_pt(data_pt)
        out_tf = model_tf(data_tf, training=False)
        out_tf = tf.log(out_tf)

        self.assertAllClose(out_pt.detach().numpy(), out_tf)


if __name__ == "__main__":
    test.main()
