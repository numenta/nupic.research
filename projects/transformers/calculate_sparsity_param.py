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


import argparse
from dataclasses import replace

# FIXME: The experiments import Ray, but it must be imported before Pickle # noqa I001
import ray  # noqa: F401, I001
from transformers import AutoModelForMaskedLM, HfArgumentParser

from experiments import CONFIGS
from nupic.research.frameworks.pytorch.model_utils import (
    calc_model_sparsity,
    count_nonzero_params,
    filter_modules,
)
from nupic.torch.modules.sparse_weights import SparseWeightsBase, rezero_weights
from run import bold, pdict
from run_args import ModelArguments
from run_utils import init_config, init_tokenizer


def calculate_sparsity_param(
    sparsity_desired,
    parameters_desired,
    experiment,
    test_sparsity=False
):
    """
    :param sparsity_desired: desired sparsity of model
    :param parameters_desired: desired number of on-params;
                               can't be used with sparsity_desired
    :param experiment: name of experiment config with a sparse architecture
    :param test_sparsity: whether to test the calculated sparsity param, this test loads
                          the model and calculates the resulting sparsity.
    """

    # Ensure sparsity_desired or parameters_desired is specified but not both.
    assert not (sparsity_desired is None and parameters_desired is None)
    assert sparsity_desired is not None or parameters_desired is not None

    print(bold("Initializing model... ") + "(this may take a minute)")
    print(f"   experiment: {experiment}")

    # Load and parse model args from config.
    exp_config = CONFIGS[experiment]
    exp_parser = HfArgumentParser(ModelArguments)
    model_args = exp_parser.parse_dict(exp_config)[0]
    model_args = replace(model_args, cache_dir=None)  # enable to run locally
    print(bold("\n\nModel parameters:\n") + pdict(model_args.__dict__))
    print()

    # Initialize model.
    config = init_config(model_args)
    tokenizer = init_tokenizer(model_args)
    model = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    print(bold("Calculating target sparsity..."))

    # Get sparse modules and calculate total number of sparsifiable params.
    sparse_modules = filter_modules(model.bert, include_modules=[SparseWeightsBase])
    sparsifiable_params = 0
    for _, m in sparse_modules.items():
        sparsifiable_params += m.zero_mask.numel()

    # Calculate the total number of params and the needed sparsity.
    total_params, _ = count_nonzero_params(model.bert)

    if parameters_desired is None:
        parameters_desired = total_params * (1 - sparsity_desired)
    elif sparsity_desired is None:
        sparsity_desired = parameters_desired / total_params

    dense_params = total_params - sparsifiable_params
    target_sparsity = 1 - (parameters_desired - dense_params) / sparsifiable_params

    print(f"   sparsity_desired: {sparsity_desired}")
    print(f"   parameters_desired: {parameters_desired}")
    print(f"   sparsifiable_params: {sparsifiable_params}")
    print(f"   total_params: {total_params}")
    print(f"   target_sparsity: {target_sparsity} (set your sparsity to this)")
    print()

    if not test_sparsity:
        return

    print(bold("Testing target sparsity..."))

    # Edit config to use the new sparsity param (sparsity=target_sparsity).
    exp_config["config_kwargs"]["sparsity"] = target_sparsity
    exp_parser = HfArgumentParser(ModelArguments)
    model_args = exp_parser.parse_dict(exp_config)[0]
    model_args = replace(model_args, cache_dir=None)  # remove to run locally

    # Initialize model; this time with the new sparsity param.
    config = init_config(model_args)
    tokenizer = init_tokenizer(model_args)
    model = AutoModelForMaskedLM.from_config(config)
    model.resize_token_embeddings(len(tokenizer))

    # Set all on-weights to one to make sure none are randomly off.
    sparse_modules = filter_modules(model.bert, include_modules=[SparseWeightsBase])
    for _, m in sparse_modules.items():
        m.weight.data[:] = 1
    model.apply(rezero_weights)  # set off weights to zero.

    resulting_sparsity = calc_model_sparsity(model.bert)
    _, nz_params = count_nonzero_params(model.bert)
    print(f"    Resulting sparsity of model.bert using sparsity={target_sparsity}\n"
          f"       actual_sparsity={resulting_sparsity}\n"
          f"       num_nonzero_params={nz_params}\n")
    print(f"    Note this may not be exactly as desired as there are "
          "discrete levels of allowable sparsity")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""This calculates what `sparsity`
param one should set for a desired level of sparsity. For instance, in some models the
desired sparsity is 80%, but not all weights, such as those for layer norm, will be
sparsified. Thus, one may need to set `sparsity=0.85` to achieve the desired 80%.
    """)
    parser.add_argument("-s", "--sparsity_desired", type=float, required=False,
                        help="Desired sparsity of BERT model.")
    parser.add_argument("-p", "--parameters_desired", type=float, required=False,
                        help="Desired on-params of BERT model. "
                        "Can't be used with `sparsity_desired`")
    parser.add_argument("-e", "--experiment", choices=list(CONFIGS.keys()),
                        help="Available experiments", required=True)
    parser.add_argument("-t", "--test_sparsity", type=bool, default=True,
                        help="Whether to test the sparsity params by loading a new "
                             "model and measuring the resulting sparsity.")

    # Ensure sparsity_desired or parameters_desired is specified, but not both.
    args = parser.parse_args()
    if args.sparsity_desired is None and args.parameters_desired is None:
        print("Must specify one of `sparsity_desired` or `parameters_desired`.")
        parser.print_help()
        exit(1)
    if args.sparsity_desired is not None and args.parameters_desired is not None:
        print("Must specify only one of `sparsity_desired` or `parameters_desired`.")
        parser.print_help()
        exit(1)

    calculate_sparsity_param(**args.__dict__)
