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

"""
Pretrained models need to be exported to be used for finetuning.
Only required argument for this script is the checkpoint folder.

Not tested for modified sparse models.
"""

import argparse

# FIXME: The experiments import Ray, but it must be imported before Pickle # noqa I001
import ray  # noqa: F401, I001
from transformers import AutoModelForMaskedLM, HfArgumentParser

from experiments import CONFIGS
from nupic.research.frameworks.pytorch.model_utils import filter_params, get_module_attr
from nupic.torch.modules.sparse_weights import SparseWeightsBase
from run_args import ModelArguments
from run_utils import init_config, init_tokenizer


def convert_to_prunable_checkpoint(checkpoint_folder, experiment):
    """
    This loads a dense models weights and a prunable model of similar architecture (one
    with SparseWeightsBase layers), copies the weights of the former into the latter,
    and then saves a new checkpoint at `{checkpoint_folder}_prunable`.

    :param checkpoint_folder: path to dense checkpoint
    :param experiment: name of experiment config with a prunable architecture
    """

    # We'll use `sparsity=0` to ensure it's dense but prunable model.
    exp_config = CONFIGS[experiment]
    exp_config["config_kwargs"]["sparsity"] = 0
    exp_parser = HfArgumentParser(ModelArguments)
    model_args = exp_parser.parse_dict(exp_config)[0]

    # Initialize prunable model and dense model.
    config = init_config(model_args)
    tokenizer = init_tokenizer(model_args)
    prunable_model = AutoModelForMaskedLM.from_config(config)
    prunable_model.resize_token_embeddings(len(tokenizer))

    dense_model = AutoModelForMaskedLM.from_pretrained(checkpoint_folder)

    # Determine which parameters belong to SparseWeightsBase classes.
    sparse_params = filter_params(prunable_model, include_modules=[SparseWeightsBase])
    sparse_dataptrs = [p.data_ptr() for p in sparse_params.values()]

    # Load the dense params into the prunable params.
    for n2, p2 in prunable_model.named_parameters():

        # e.g. replace `linear.module.weight` with `linear.weight` when appropriate.
        if p2.data_ptr() in sparse_dataptrs:
            n1 = n2.replace(".module", "")
        else:
            n1 = n2

        p1 = get_module_attr(dense_model, n1)
        p2.data[:] = p1

    # Save the prunable model.
    new_folder_name = checkpoint_folder + "_prunable"
    prunable_model.save_pretrained(new_folder_name)
    print(f"Saved prunable model to:\n{new_folder_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_folder", type=str,
                        help="Path to checkpoint to convert")
    parser.add_argument("-e", "--experiment", choices=list(CONFIGS.keys()),
                        help="Available experiments", required=True)

    args = parser.parse_args()
    convert_to_prunable_checkpoint(**args.__dict__)
