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

import argparse
import os
from ast import literal_eval
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from ray import tune


def save_plot(expdir, outfilename, show_expected=False, mean=False):
    analysis = tune.Analysis(expdir)
    configs = analysis.get_all_configs()

    layernames = ("cnn1", "cnn2", "fc1", "fc2",)

    nz_by_unit_key = ("expected_nz_by_unit" if show_expected
                      else "inference_nz_by_unit")

    f = plt.figure(figsize=(12, 5))
    ax = f.add_subplot(111)
    ax.yaxis.tick_right()

    data_by_l0 = defaultdict(list)

    for trial_path, df in analysis.trial_dataframes.items():
        accuracies = []
        nz_counts = []

        for epoch in range(len(df)):
            nz = 0
            for layername in layernames:
                nz_by_unit = literal_eval(
                    df.at[epoch, "{}/{}".format(layername,
                                                nz_by_unit_key)])
                nz += np.sum(nz_by_unit)
            accuracies.append(df.at[epoch, "mean_accuracy"])
            nz_counts.append(nz)

        l0_strength = configs[trial_path]["l0_strength"]

        data_by_l0[l0_strength].append((np.array(accuracies),
                                        np.array(nz_counts)))

    if mean:
        new = {}
        for l0_strength, results in data_by_l0.items():
            all_accuracies, all_nz_counts = zip(*results)
            new[l0_strength] = [(
                np.mean(all_accuracies, axis=0),
                np.mean(all_nz_counts, axis=0)
            )]
        data_by_l0 = new

    colors = dict(zip(data_by_l0.keys(), ("C0", "C1", "C2", "C3",)))

    for l0_strength, results in data_by_l0.items():
        color = colors[l0_strength]
        for accuracies, nz_counts in results:
            ax.plot(accuracies, nz_counts, "-o", markersize=2, color=color)

    ax.set_ylim(0, ax.get_ylim()[1])

    outpath = os.path.join(expdir, outfilename)
    print("Saving {}".format(outpath))
    plt.savefig(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", type=str,
                        help="Path to a ray tune experiment")
    parser.add_argument("--outfilename", type=str,
                        default="accuracy_sparsity_during_training.pdf")
    parser.add_argument("--mean", action="store_true")
    parser.add_argument("--expected", action="store_true")
    args = parser.parse_args()
    save_plot(args.expdir, args.outfilename, show_expected=args.expected,
              mean=args.mean)
