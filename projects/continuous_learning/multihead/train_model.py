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
This module provides example code to train a dense/sparse MLP in a continuous learning
setup using the multihead setting.
"""

import argparse

import torch

from model import Classifier
from nupic.research.frameworks.continuous_learning.multihead.multihead import (
    do_training,
)
from nupic.torch.modules import rezero_weights


def post_batch(model):
    model.apply(rezero_weights)


if __name__ == "__main__":

    # identify training scenario, dataset, and model sparsity
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="task",
                        choices=["task", "domain", "class"])
    parser.add_argument("--dataset", type=str, default="splitGSC",
                        choices=["splitMNIST", "splitGSC"])
    parser.add_argument("--sparse", action="store_true",
                        help="train a sparse classifier instead of a dense one")
    args = parser.parse_args()

    # shape of the input to the classifier
    input_sizes = {
        "splitMNIST": 28 * 28,
        "splitGSC": 32 * 32
    }

    # initialize classifier
    model = Classifier(is_sparse=args.sparse, input_size=input_sizes[args.dataset])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training loop
    do_training(model, args.dataset, args.scenario, device,
                post_batch_callback=post_batch if args.sparse else None)
