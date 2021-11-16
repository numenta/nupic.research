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
"""
This script is used to import GSC models pre-trained in pytorch into tensorflow
"""

import torch.hub

from nupic.research.frameworks.tensorflow.utils import load_gsc_weights_from_pytorch
from nupic.tensorflow.models import GSCSparseCNN, GSCSuperSparseCNN

if __name__ == "__main__":
    print("Loading pre-trained gsc_sparse_cnn from pytorch hub")
    model_pt = torch.hub.load(github="numenta/nupic.torch",
                              model="gsc_sparse_cnn",
                              progress=True,
                              pretrained=True)

    print("Converting gsc_sparse_cnn from pytorch to tensorflow")
    model_tf = GSCSparseCNN(data_format="channels_last")
    load_gsc_weights_from_pytorch(model_tf, model_pt)
    model_tf.compile(optimizer="sgd",
                     loss="sparse_categorical_crossentropy",
                     metrics=["accuracy"])
    print("Saving pre-trained tensorflow version of gsc_sparse_cnn as "
          "gsc_sparse_cnn.h5")
    model_tf.save_weights("gsc_sparse_cnn.h5")

    print("Loading pre-trained gsc_super_sparse_cnn from pytorch hub")
    model_pt = torch.hub.load(github="numenta/nupic.torch",
                              model="gsc_super_sparse_cnn",
                              progress=True,
                              pretrained=True)
    print("Converting gsc_super_sparse_cnn from pytorch to tensorflow")
    model_tf = GSCSuperSparseCNN(data_format="channels_last")
    load_gsc_weights_from_pytorch(model_tf, model_pt)
    print("Saving pre-trained tensorflow version of gsc_super_sparse_cnn as "
          "gsc_super_sparse_cnn.h5")
    model_tf.save_weights("gsc_super_sparse_cnn.h5")
