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
from __future__ import print_function

import click
from tabulate import tabulate

# Constants values used across all experiments
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
    default="experiments.cfg",
    show_default=True,
    help="your experiments config file",
)
@click.option(
    "-e",
    "--experiment",
    multiple=True,
    help="print only selected experiments, by default run all "
    "experiments in config file.",
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
    configs = parse_config(config, experiment, globals_param=globals())
    if show_list:
        print("Experiments:", list(configs.keys()))
        return

    params_table = [
        [
            "Network",
            "L1 F",
            "L1 Sparsity",
            "L2 F",
            "L2 Sparsity",
            "L3 N",
            "L3 Sparsity",
            "Wt Sparsity",
        ]
    ]
    for name, params in configs.items():
        linear_n = params["linear_n"]
        linear_percent_on = params["linear_percent_on"]
        weight_sparsity = params["weight_sparsity"]
        cnn_percent_on = params["cnn_percent_on"]
        cnn_out_channels = params["cnn_out_channels"]

        l3_n = linear_n[0]
        l3_sp = "{0:.1f}%".format(100 * linear_percent_on[0])
        wt_sp = "{0}%".format(100 * weight_sparsity)

        if len(cnn_percent_on) == 2:
            l1_percent_on = cnn_percent_on[0]
            l2_percent_on = cnn_percent_on[1]
        else:
            l1_percent_on = cnn_percent_on[0]
            l2_percent_on = None

        if len(cnn_out_channels) == 2:
            l1_f = cnn_out_channels[0]
            l1_sp = "{0:.1f}%".format(100 * l1_percent_on)

            # Feed CNN-1 output to CNN-2
            l2_f = cnn_out_channels[1]
            l2_sp = "{0:.1f}%".format(100 * l2_percent_on)
        else:
            l1_f = cnn_out_channels[0]
            l1_sp = "{0:.1f}%".format(100 * l1_percent_on)

            l2_f = None
            l2_sp = None

        params_table.append([name, l1_f, l1_sp, l2_f, l2_sp, l3_n, l3_sp, wt_sp])

    print()
    print(tabulate(params_table, headers="firstrow", tablefmt=tablefmt))


if __name__ == "__main__":
    main()
