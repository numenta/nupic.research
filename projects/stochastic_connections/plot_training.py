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
import glob
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_plot(folderpath, outpath, show_expected=False):
    steps = []

    for filepath in sorted(glob.glob(str(folderpath / "res*.pkl"))):
        with open(filepath, "rb") as f:
            steps.append(pickle.load(f))

    layernames = ("cnn1", "cnn2", "fc1", "fc2",)

    w = len(layernames) * 3
    h = len(steps) * 2

    fig, axes = plt.subplots(len(steps), len(layernames), figsize=(w, h))

    nz_by_unit_key = ("expected_nz_by_unit" if show_expected
                      else "inference_nz_by_unit")

    for t, (_results, nonzeros) in enumerate(steps):
        for i, layername in enumerate(layernames):
            ax = axes[t, i]
            num_input_units = nonzeros[layername]["num_input_units"]
            ax.hist(nonzeros[layername][nz_by_unit_key],
                    bins=np.arange(0, num_input_units + 1,
                                   max(num_input_units / 50, 1)))

    for ax, col in zip(axes[0], layernames):
        ax.set_title(col)

    for ax, row in zip(axes[:, 0], range(len(steps))):
        ax.set_ylabel("Epoch {}".format(row), size="large")

    fig.suptitle("# nonzero weights per unit", y=1.01)
    plt.tight_layout()

    print("Saving {}".format(outpath))
    plt.savefig(outpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("foldername", type=str)

    args = parser.parse_args()

    cwd = Path(os.path.dirname(os.path.realpath(__file__)))
    folderpath = cwd / args.foldername

    save_plot(folderpath, folderpath / "timesteps.pdf")
