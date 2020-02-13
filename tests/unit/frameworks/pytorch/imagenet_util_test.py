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

from nupic.research.frameworks.pytorch.imagenet.experiment_utils import create_model
from nupic.research.frameworks.pytorch.model_compare import compare_models
from nupic.research.frameworks.pytorch.model_utils import serialize_state_dict
from nupic.research.frameworks.pytorch.models.resnets import resnet50


class ImagenetExperimentUtilsTest(unittest.TestCase):

    def test_creaate_model_from_checkpoint(self):
        model1 = create_model(model_class=resnet50, model_args={},
                              init_batch_norm=False, device="cpu")

        # Simulate imagenet experiment by changing the weights
        def init(m):
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.data.fill_(0.042)
        model1.apply(init)

        # Save model checkpoint only, ignoring optimizer and other imagenet
        # experiment objects state. See ImagenetExperiment.get_state
        state = {}
        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, model1.state_dict())
            state["model"] = buffer.getvalue()

        with tempfile.NamedTemporaryFile() as checkpoint_file:
            # Ray save checkpoints as pickled dicts
            pickle.dump(state, checkpoint_file)
            checkpoint_file.file.flush()

            # Load model from checkpoint
            model2 = create_model(
                model_class=resnet50, model_args={},
                init_batch_norm=False, device="cpu",
                checkpoint_file=checkpoint_file.name)

        self.assertTrue(compare_models(model1, model2, (3, 32, 32)))
