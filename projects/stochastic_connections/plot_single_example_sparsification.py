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

import matplotlib.pyplot as plt
import numpy as np
from ray import tune


def save_plot(analysis, outfilename, show_expected=False):
    layernames = ("cnn1", "cnn2", "fc1", "fc2",)
    nz_by_unit_key = ("expected_nz_by_unit" if show_expected
                      else "inference_nz_by_unit")

    for trial_path, df in analysis.trial_dataframes.items():
        epochs = len(df)

        w = len(layernames) * 3
        h = epochs * 2
        fig, axes = plt.subplots(epochs, len(layernames), figsize=(w, h))

        for epoch in range(epochs):
            for i, layername in enumerate(layernames):
                ax = axes[epoch, i]
                num_nonzeros = literal_eval(df.at[
                    epoch, "{}/{}".format(layername, nz_by_unit_key)])
                num_input_units = df.at[epoch,
                                        "{}/num_input_units".format(layername)]
                ax.hist(num_nonzeros,
                        bins=np.arange(0, num_input_units + 1,
                                       max(num_input_units / 50, 1)))

        for ax, col in zip(axes[0], layernames):
            ax.set_title(col)

        for ax, row in zip(axes[:, 0], range(epochs)):
            ax.set_ylabel("Epoch {}".format(row), size="large")

        fig.suptitle("# nonzero weights per unit", y=1.01)
        plt.tight_layout()

        outpath = os.path.join(trial_path, outfilename)
        print("Saving {}".format(outpath))
        plt.savefig(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str,
                        help="Path to a single trial of a ray tune experiment")
    parser.add_argument("--outfilename", type=str, default="sparsity_log.pdf")
    parser.add_argument("--expected", action="store_true")
    args = parser.parse_args()
    save_plot(tune.Analysis(args.logdir), args.outfilename, args.expected)
