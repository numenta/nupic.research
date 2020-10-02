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
Quantize resnet50 model weights only, mimicking pytorch quantization
"""

import argparse
import copy

import torch
from torch import nn

from experiments import CONFIGS
from nupic.research.frameworks.pytorch.model_utils import evaluate_model
from nupic.torch.modules.sparse_weights import SparseWeightsBase


def fold_batchnorm_conv_(conv2d, bn_2d):
    """
    Mutate conv2d in place, folding a batchnorm into it.
    """
    if bn_2d.affine:
        bn_w = bn_2d.weight
        bn_b = bn_2d.bias
    else:
        bn_w = torch.ones(bn_2d.num_features)
        bn_b = torch.zeros(bn_2d.num_features)

    t = (bn_2d.running_var + bn_2d.eps).rsqrt()
    conv2d.weight.data[:] = conv2d.weight.detach() * (bn_w * t).reshape((-1, 1, 1, 1))

    assert conv2d.bias is None
    conv2d.bias = nn.Parameter((-bn_2d.running_mean) * t * bn_w + bn_b)


def minmax_symmetric_quantize(weight, min_vals, max_vals):
    """
    Mimic pytorch's _ObserverBase.per_channel_symmetric quantization
    """
    qmax = 127
    qmin = -128

    zero_points = torch.zeros(min_vals.size(), dtype=torch.int64)
    if torch.equal(max_vals, min_vals):
        scales = torch.ones(min_vals.size(), dtype=torch.float)
    else:
        max_vals = torch.max(-min_vals, max_vals)
        scales = max_vals / ((qmax - qmin) / 2)
        scales = torch.max(scales, torch.tensor([1e-8],
                                                device=scales.device,
                                                dtype=scales.dtype))

    return torch.quantize_per_channel(weight.data.cpu(), scales.cpu(),
                                      zero_points, axis=0, dtype=torch.qint8)


def emulate_quantized_conv_weights(conv2d):
    """
    Quantize then dequantize the weights.
    """
    min_vals, _ = conv2d.weight.detach().min(dim=1)
    min_vals, _ = min_vals.min(dim=1)
    min_vals, _ = min_vals.min(dim=1)
    min_vals.clamp_(max=0.)

    max_vals, _ = conv2d.weight.detach().max(dim=1)
    max_vals, _ = max_vals.max(dim=1)
    max_vals, _ = max_vals.max(dim=1)
    max_vals.clamp_(min=0.)

    w = minmax_symmetric_quantize(conv2d.weight, min_vals, max_vals)
    conv2d.weight.data[:] = w.dequantize()


def emulate_quantized_linear_weights(linear):
    """
    Quantize then dequantize the weights.
    """
    min_vals, _ = linear.weight.detach().min(dim=1)
    min_vals.clamp_(max=0.)
    max_vals, _ = linear.weight.detach().max(dim=1)
    max_vals.clamp_(min=0.)

    w = minmax_symmetric_quantize(linear.weight, min_vals, max_vals)
    linear.weight.data[:] = w.dequantize()


def get_sparseweights_module(module):
    if isinstance(module, SparseWeightsBase):
        return module.module
    else:
        return module


def emulate_quantized_resnet_weights(model):
    """
    Fuse the batchnorm and conv layers, then quantize and dequantize the weights
    in every conv and linear layer. This simulates a weights-only quantization.
    """
    model = copy.deepcopy(model)

    conv2d = get_sparseweights_module(model.features[0])
    fold_batchnorm_conv_(conv2d, model.features[1])
    emulate_quantized_conv_weights(conv2d)
    del model.features[1]
    for group in ["group1", "group2", "group3", "group4"]:
        for block in getattr(model.features, group):
            for conv_idx, bn_idx in [(0, 1), (3, 4), (6, 7)]:
                conv2d = get_sparseweights_module(block.regular_path[conv_idx])
                fold_batchnorm_conv_(conv2d, block.regular_path[bn_idx])
                emulate_quantized_conv_weights(conv2d)

            del block.regular_path[7]
            del block.regular_path[4]
            del block.regular_path[1]
            if isinstance(block.shortcut, nn.Sequential):
                conv2d = get_sparseweights_module(block.shortcut[0])
                fold_batchnorm_conv_(conv2d, block.shortcut[1])
                emulate_quantized_conv_weights(conv2d)
                del block.shortcut[1]

    linear = get_sparseweights_module(model.classifier)
    emulate_quantized_linear_weights(linear)

    return model


def main(args):
    # Get experiment configuration
    config = copy.deepcopy(CONFIGS[args.name])
    config.update(vars(args))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config["device"] = device
    experiment_class = config["experiment_class"]
    model = experiment_class.create_model(config=config, device=device)
    qmodel = emulate_quantized_resnet_weights(model)

    val_loader = experiment_class.create_validation_dataloader(config)
    loss_function = config.get("loss_function", torch.nn.functional.cross_entropy)
    validate_function = config.get("evaluate_model_func", evaluate_model)
    result = validate_function(qmodel, val_loader, device,
                               criterion=loss_function,
                               progress={
                                   "desc": "testing (int8 weights)",
                                   "leave": False
                               })
    actual = result["mean_accuracy"]

    print(f"experiment: {args.name}, after: {actual}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-e", "--experiment", dest="name", help="Experiment to run",
        choices=CONFIGS.keys(),
    )
    parser.add_argument(
        "--checkpoint-file",
        dest="checkpoint_file",
        help="Checkpoint to load",
    )
    parser.add_argument(
        "-w", "--workers", default=4, type=int,
        help="Number of dataloader workers"
    )

    args = parser.parse_args()
    if args.name is None:
        parser.print_help()
        exit(1)

    main(args)
