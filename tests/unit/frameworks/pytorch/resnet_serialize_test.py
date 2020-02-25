# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import unittest

import tempfile
import torch

from nupic.research.frameworks.pytorch.models.resnets import ResNet
from nupic.research.frameworks.pytorch.imagenet.experiment_utils import create_model
from nupic.research.frameworks.pytorch.model_compare import compare_models


class ResNetSerialization(unittest.TestCase):
    '''Test if loaded model is identical to initially saved one'''
    def test_identical(self):
        # model args for ResNet, may become function arguments later on
        model_args = dict(config=dict(
            num_classes=3,
            defaults_sparse=True,
        ))
        model_class = ResNet
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = create_model(
            model_class=model_class,
            model_args=model_args,
            init_batch_norm=False,
            device=device,
        )
        # create temp file
        model_path = tempfile.NamedTemporaryFile(delete=True)
        # and write the created model
        torch.save(model.state_dict(), model_path)

        model_copy = create_model(
            model_class=model_class,
            model_args=model_args,
            init_batch_norm=False,
            device=device,
        )
        # copy the initial model parameters to the new model
        model_copy.load_state_dict(torch.load(model_path.name))

        # comapre the initial and new model
        self.assertTrue(compare_models(model, model_copy, input_shape=(3, 224, 224)))
        model_path.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
