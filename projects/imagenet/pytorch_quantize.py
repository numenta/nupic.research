# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
Quantize resnet50 models using pytorch quantization tools
"""

import argparse
import copy
import io
import os

import torch

from experiments import CONFIGS
from nupic.research.frameworks.pytorch.model_utils import (
    evaluate_model,
    set_random_seed,
)


def print_size_of_model(model):
    with io.BytesIO() as buffer:
        torch.save(model.state_dict(), buffer)
        print(f"Size (bytes): {len(buffer.getvalue()):,}")


def quantize(config):
    """
    Quantize model using pytorch static quantization functions
    See https://pytorch.org/blog/introduction-to-quantization-on-pytorch/#post-training-static-quantization  # noqa: E501

    :param config: Imagenet Experiment configuration pre-configured with a valid
                   checkpoint file of the model to quantize
    """
    backend = config.pop("backend")
    torch.backends.quantized.engine = backend

    # Create model from the experiment
    experiment_name = config["name"]
    device = config["device"]
    experiment_class = config["experiment_class"]

    model = experiment_class.create_model(config=config, device=device)
    print(model)

    # Quantize model
    q_model = copy.deepcopy(model)
    q_model.cpu()
    q_model.eval()

    # Fuse Conv and bn
    q_model.fuse_model(fuse_relu=True)

    # Specify quantization configuration
    q_model.qconfig = torch.quantization.get_default_qconfig(backend)
    print("qconfig:", q_model.qconfig)

    # Prepare model for quantization
    torch.quantization.prepare(q_model, inplace=True)

    # Get validation function and data loaders from the experiment
    train_loader = experiment_class.create_train_dataloader(config)
    loss_function = config.get("loss_function", torch.nn.functional.cross_entropy)
    validate_function = config.get("evaluate_model_func", evaluate_model)

    # Calibrate using a few batches from the training dataset
    calibration = config.get("calibration", 10)
    validate_function(
        model=q_model, loader=train_loader, device="cpu", criterion=loss_function,
        batches_in_epoch=calibration,
        progress={"desc": "calibrating", "leave": False})

    # Convert to quantized model
    q_model = torch.quantization.convert(q_model, inplace=True)
    print(q_model)

    print("=" * 40)
    print(f"Quantization config: {q_model.qconfig}")

    # Print quantized results
    if config.get("test", False):
        val_loader = experiment_class.create_validation_dataloader(config)
        results = validate_function(
            model=q_model, loader=val_loader, device="cpu", criterion=loss_function,
            progress={"desc": "testing (int8)",
                      "leave": False},
        )

        print("=" * 40)
        print(f"Baseline ({experiment_name})")
        print_size_of_model(model)

        print("=" * 40)
        accuracy = results["mean_accuracy"]
        print(f"Quantized ({experiment_name}): {accuracy}")
        print_size_of_model(q_model)
        print("=" * 40)

    return q_model


def main(args):
    # Get experiment configuration
    config = copy.deepcopy(CONFIGS[args.name])
    config.update(vars(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device

    # Replace dynamic seed (i.e. 'tune.sample_from') with constant
    seed = config.get("seed", 42)
    if not isinstance(seed, int):
        seed = 42
        config["seed"] = seed
    set_random_seed(seed)

    q_model = quantize(config)

    # Save quantized model
    output_file_name = os.path.join(args.output, f"{args.name}.{args.backend}.pt")
    print(f"Saving quantized model '{args.name}' weights to '{output_file_name}'")
    torch.jit.save(torch.jit.script(q_model), output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-e",
        "--experiment",
        dest="name",
        help="Experiment to run",
        choices=CONFIGS.keys(),
    )
    parser.add_argument(
        "--checkpoint-file",
        dest="checkpoint_file",
        help="Checkpoint to load",
    )
    parser.add_argument(
        "-w", "--workers", default=4, type=int, help="Number of dataloader workers"
    )
    parser.add_argument(
        "-o", "--output", default=os.getcwd(), help="Quantized model destination"
    )
    parser.add_argument(
        "-b",
        "--backend",
        choices=["qnnpack", "fbgemm"],
        help="Pytorch Quantization backend",
        default="fbgemm",
    )
    parser.add_argument(
        "-c", "--calibration", default=10, help="Number of calibration batches"
    )

    parser.add_argument(
        "-t", "--test", action="store_true",
        help="Whether or not to test quantized model"
    )

    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
        exit(1)

    main(args)
