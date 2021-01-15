#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
"""
Use this script to analyze GSC experiments
"""
import argparse

from tabulate import tabulate

from experiments import CONFIGS
from nupic.research.frameworks.pytorch.models import LeSparseNet
from nupic.torch.models import GSCSparseCNN


def print_parameters_table(experiments, tablefmt):
    """
    Print GSC parameters table
    """
    table_1 = [[
        "Network",
        "L1 Filters",
        "L1 Act Sparsity",
        "L1 Wt Sparsity",
        "L2 Filters",
        "L2 Act Sparsity",
        "L2 Wt Sparsity",
        "L3 N",
        "L3 Act Sparsity",
        "Wt Sparsity",
    ]]
    table_2 = [[
        "Network",
        "L1 Activation Sparsity",
        "L2 Activation Sparsity",
        "L3 Activation Sparsity",
    ]]
    table_3 = [[
        "Network",
        "L1 Weight Sparsity",
        "L2 Weight Sparsity",
        "L3 Weight Sparsity",
    ]]

    for name in experiments:
        config = CONFIGS[name]
        experiment_class = config["experiment_class"]
        model = experiment_class.create_model(config, device="cpu")

        l1_f, l1_sp, l1_wt_sp, l2_f, l2_sp, l2_wt_sp, l3_n, l3_sp, l3_wt_sp = [0] * 9
        if isinstance(model, GSCSparseCNN):
            l1_f = model.cnn1.module.out_channels
            l1_wt_sp = model.cnn1.weight_sparsity
            l1_sp = model.cnn1_kwinner.percent_on

            l2_f = model.cnn2.module.out_channels
            l2_sp = model.cnn2_kwinner.percent_on
            l2_wt_sp = model.linear.weight_sparsity

            l3_sp = model.linear_kwinner.percent_on
            l3_n = model.linear.module.out_features
            l3_wt_sp = model.linear.weight_sparsity
        elif isinstance(model, LeSparseNet):
            l1_f = model.cnn1_cnn.module.out_channels
            l1_wt_sp = model.cnn1_cnn.weight_sparsity
            l1_sp = model.cnn1_kwinner.percent_on

            l2_f = model.cnn2_cnn.module.out_channels
            l2_wt_sp = model.cnn2_cnn.weight_sparsity
            l2_sp = model.cnn2_kwinner.percent_on

            l3_sp = model.linear1_kwinners.percent_on
            l3_n = model.linear1.module.out_features
            l3_wt_sp = model.linear1.weight_sparsity

        table_1.append([name, l1_f, l1_sp, l1_wt_sp, l2_f, l2_sp, l2_wt_sp,
                        l3_n, l3_sp, l3_wt_sp])
        table_2.append([name, l1_f, l2_f, l3_n])
        table_3.append([name, l1_wt_sp, l2_wt_sp, l3_wt_sp])

    print()
    print(tabulate(table_1, headers="firstrow", tablefmt=tablefmt,
                   floatfmt="0.1%", stralign="center", ))

    print()
    print(tabulate(table_2, headers="firstrow", tablefmt=tablefmt,
                   stralign="left", numalign="center"))

    print()
    print(tabulate(table_3, headers="firstrow", tablefmt=tablefmt,
                   floatfmt="0.1%", stralign="left", numalign="center"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument("-e", "--experiments", dest="experiments",
                        nargs="+",
                        choices=list(CONFIGS.keys()),
                        help="Experiments to analyze",
                        default=list(CONFIGS.keys()))
    parser.add_argument("--tablefmt", default="grid",
                        choices=["plain", "simple", "grid", "pipe", "orgtbl",
                                 "rst", "mediawiki", "latex", "latex_raw",
                                 "latex_booktabs"],
                        help="Table format")
    args = parser.parse_args()
    print_parameters_table(args.experiments, args.tablefmt)
