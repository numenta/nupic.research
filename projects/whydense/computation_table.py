# -*- coding: utf-8 -*-
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
import click
import numpy as np
from tabulate import tabulate

from nupic.research.support import parse_config

STRIDE = 1
PADDING = 0
KERNEL_SIZE = 5


@click.command()
@click.option(
    "-c",
    "--config",
    metavar="FILE",
    type=open,
    default="gsc/experiments.cfg",
    show_default=True,
    help="your experiments config file",
)
@click.option(
    "-e",
    "--experiment",
    default=["denseCNN2", "sparseCNN2"],
    multiple=True,
    help="Selected 2 experiments to compare.",
)
@click.option(
    "-f",
    "--format",
    "tablefmt",
    help="Table format",
    type=click.Choice(choices=["grid", "latex"]),
    show_default=True,
    default="grid",
)
@click.option(
    "-l",
    "--list",
    "show_list",
    is_flag=True,
    help="show list of available experiments.",
)
def main(config, experiment, tablefmt, show_list):
    assert len(experiment) == 2, "Select 2 experiments (denseCNN2, sparseCNN2)"

    configs = parse_config(config, experiment, globals_param=globals())
    if show_list:
        print("Experiments:", list(configs.keys()))
        return

    # Sort with dense configurations first
    configs = sorted(configs.items(),
                     key=lambda x: 0 if x[0].lower().startswith("dense") else 1)

    params_table = [
        [
            "Network",
            "L1",
            "L2",
            "L3",
            "Output",
            "Total"
        ]
    ]
    l1_ratio = l2_ratio = l3_ratio = output_ratio = total_ratio = 1.0
    for name, params in configs:
        input_shape = params["input_shape"]
        input_c, height, width = input_shape

        # CNN configuration
        cnn_out_channels = params["cnn_out_channels"]
        cnn_percent_on = params["cnn_percent_on"]
        cnn_weight_sparsity = params["cnn_weight_sparsity"]

        # Linear configuration
        linear_n = params["linear_n"]
        linear_percent_on = params["linear_percent_on"]
        weight_sparsity = params["weight_sparsity"]

        # Compute total non-zero weights in L1
        l1_out_c = cnn_out_channels[0]
        l1_w = input_c * l1_out_c * KERNEL_SIZE * KERNEL_SIZE
        l1_w = l1_w * cnn_weight_sparsity[0]

        # Input density is 1, so L1 multiplies = output_shape * L1 weights
        l1_out_width = (width - KERNEL_SIZE + 1)
        l1_out_height = (height - KERNEL_SIZE + 1)
        l1_mul = l1_out_width * l1_out_height * l1_w

        # L1 Output after maxpool
        l1_out = [l1_out_c, l1_out_height / 2, l1_out_width / 2]

        # Compute total non-zero weights in L2
        l2_out_c = cnn_out_channels[1]
        l2_w = l1_out_c * l2_out_c * KERNEL_SIZE * KERNEL_SIZE
        l2_w = l2_w * cnn_weight_sparsity[1]

        # L2 multiplies = input_sparsity * L2 weights * L2 output size
        l2_out_height = (l1_out[1] - KERNEL_SIZE + 1)
        l2_out_width = (l1_out[2] - KERNEL_SIZE + 1)
        l2_mul = cnn_percent_on[0] * l2_out_height * l2_out_width * l2_w

        # L2 Output after pool
        l2_out = [l2_out_c, l2_out_height / 2, l2_out_width / 2]

        # Compute total non-zero weights in L3
        l3_w = np.prod(l2_out) * linear_n[0]
        l3_w = l3_w * weight_sparsity[0]

        # L3 multiplies = l2 sparsity * L3 weights
        l3_mul = cnn_percent_on[1] * l3_w

        # L3 Output
        l3_out = linear_n[0]
        l3_nnz_out = l3_out * linear_percent_on[0]

        # Output layer multiplies = l3 non-zero output * weights
        output_w = l3_out * params["num_classes"]
        output_mul = l3_nnz_out * output_w

        # Compute gain ratio against previous configuration
        l1_ratio = l1_mul / l1_ratio
        l2_ratio = l2_mul / l2_ratio
        l3_ratio = l3_mul / l3_ratio
        output_ratio = output_mul / output_ratio
        total_mul = l1_mul + l2_mul + l3_mul + output_mul
        total_ratio = total_mul / total_ratio

        params_table.append([name,
                             "{:,.0f}".format(l1_mul),
                             "{:,.0f}".format(l2_mul),
                             "{:,.0f}".format(l3_mul),
                             "{:,.0f}".format(output_mul),
                             "{:,.0f}".format(total_mul),
                             ])

    params_table.append(["Computation Efficiency",
                         "{:.0f} x".format(1.0 / l1_ratio),
                         "{:.0f} x".format(1.0 / l2_ratio),
                         "{:.0f} x".format(1.0 / l3_ratio),
                         "{:.0f} x".format(1 / output_ratio),
                         "{:.0f} x".format(1 / total_ratio),
                         ])

    print(tabulate(params_table, headers="firstrow", tablefmt=tablefmt,
                   stralign="center", floatfmt=",.0f"))


if __name__ == "__main__":
    main()
