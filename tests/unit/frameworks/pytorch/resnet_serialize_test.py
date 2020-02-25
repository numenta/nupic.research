#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
import pickle
import tempfile
import unittest

import torch

import nupic.research.frameworks.pytorch.models.resnets
from nupic.research.frameworks.pytorch.imagenet.experiment_utils import create_model
from nupic.research.frameworks.pytorch.model_compare import compare_models
from nupic.research.frameworks.pytorch.model_utils import serialize_state_dict


def save_model(model):
    state = {}
    with io.BytesIO() as buffer:
        serialize_state_dict(buffer, model.state_dict())
        state["model"] = buffer.getvalue()

    checkpoint_path = tempfile.NamedTemporaryFile(delete=True)
    pickle.dump(state, checkpoint_path)
    checkpoint_path.flush()
    return checkpoint_path


class ResNetSerialization(unittest.TestCase):
    def test_identical(self):
        model_args = dict(config=dict(
            num_classes=3,
            defaults_sparse=True,
        ))
        model_class = nupic.research.frameworks.pytorch.models.resnets.resnet50
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = create_model(
            model_class=model_class,
            model_args=model_args,
            init_batch_norm=False,
            device=device,
        )

        checkpoint_file = save_model(model)

        model2 = create_model(
            model_class=model_class,
            model_args=model_args,
            init_batch_norm=False,
            device=device,
            checkpoint_file=checkpoint_file.name
        )

        checkpoint_file.close()
        assert checkpoint_file.file.closed
        self.assertTrue(compare_models(model, model2, (3, 224, 224)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
