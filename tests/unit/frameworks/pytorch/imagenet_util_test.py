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
# import subprocess
import tempfile
import unittest
from pathlib import Path

from nupic.research.frameworks.pytorch.imagenet.network_utils import create_model
from nupic.research.frameworks.pytorch.model_compare import compare_models
from nupic.research.frameworks.pytorch.model_utils import (
    serialize_state_dict,
    set_random_seed,
)
from nupic.research.frameworks.pytorch.models.resnets import resnet50
from nupic.torch.modules import rezero_weights

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"

TEST_MODEL_ARGS = dict(config=dict(
    activation_params_func=lambda *_: dict(percent_on=0.5, local=True),
    # Old SparseWeight class keeps a buffer with all non-zero indices
    # Use low sparsity (high density) values to create sparse models with
    # smaller checkpoints
    conv_params_func=lambda *_: dict(sparsity=0.01),
    linear_params_func=lambda *_: dict(sparsity=0.01)
))


def _create_test_model(checkpoint_file=None):
    """
    Create standard resnet50 model to be used in tests.
    """
    model = create_model(model_class=resnet50, model_args=TEST_MODEL_ARGS,
                         init_batch_norm=False, checkpoint_file=checkpoint_file,
                         device="cpu")

    # Simulate imagenet experiment by changing the weights
    def init(m):
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.fill_(0.042)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0.0)

    if checkpoint_file is None:
        model.apply(init)
        model.apply(rezero_weights)

    return model


def _create_test_checkpoint(file_name):
    """
    Creates a checkpoint file to be used with `test_checkpoint_backward_compatibility`.
    Whenever `test_checkpoint_backward_compatibility` test fails you need to
    create a new checkpoint file from the previous version (commit) using this
    function and update the test to include the new file
    """
    model = _create_test_model()

    # Save model checkpoint only, ignoring optimizer and other imagenet
    # experiment objects state. See ImagenetExperiment.get_state
    state = {}
    with io.BytesIO() as buffer:
        serialize_state_dict(buffer, model.state_dict(), compresslevel=9)
        state["model"] = buffer.getvalue()

    with open(file_name, "wb") as checkpoint_file:
        pickle.dump(state, checkpoint_file)
        checkpoint_file.flush()


class ImagenetExperimentUtilsTest(unittest.TestCase):
    def setUp(self):
        set_random_seed(42)

    def test_creaate_model_from_checkpoint(self):
        model1 = _create_test_model()

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
                model_class=resnet50, model_args=TEST_MODEL_ARGS,
                init_batch_norm=False, device="cpu",
                checkpoint_file=checkpoint_file.name)

        self.assertTrue(compare_models(model1, model2, (3, 32, 32)))

    @unittest.skip("This test should not rely on the random number generator. "
                   "Instead, it should replace all SparseWeights masks with "
                   "a fixed pattern, for example a zero at every 99th weight. "
                   "The test is currently broken because the SparseWeights "
                   "uses the random number generator differently than it did "
                   "when this test was written")
    def test_checkpoint_backward_compatibility(self):
        current_model = _create_test_model()

        # Get checkpoint prior to resnet naming changes (commit d1e8cad)
        model_d1e8cad = _create_test_model(
            checkpoint_file=CHECKPOINTS_DIR / "checkpoint_d1e8cad.pt")

        self.assertTrue(compare_models(current_model, model_d1e8cad, (3, 32, 32)))

        # Get checkpoint after resnet naming changes (commit 91ee855)
        model_91ee855 = _create_test_model(
            checkpoint_file=CHECKPOINTS_DIR / "checkpoint_91ee855.pt")
        self.assertTrue(compare_models(current_model, model_91ee855, (3, 32, 32)))


if __name__ == "__main__":
    # Create new checkpoint file for `test_checkpoint_backward_compatibility`
    # Before running this script, make sure checkout to the latest working version
    # result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
    #                         capture_output=True)
    # sha = result.stdout.decode().strip()
    # CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    # set_random_seed(42)
    # _create_test_checkpoint(CHECKPOINTS_DIR / f"checkpoint_{sha}.pt")
    unittest.main(verbosity=2)
