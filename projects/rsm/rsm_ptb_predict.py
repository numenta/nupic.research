#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import argparse
import os
import pickle
from functools import partial

import torch

import lang_util
from nupic.research.support.parse_config import parse_config
from rsm_experiment import RSMExperiment

# Fix for UnicodeDecodeError in torch.load (maybe?)
pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

batch_size = 10
n_batches = 10


def show_predictions(config):
    """
    Run small batches from validation set and show prediction results
    """
    exp = RSMExperiment(config=config)
    exp.model_setup(config)

    exp.model_restore(config.get("checkpoint"))

    corpus = lang_util.Corpus(config.get("data_dir") + "/PTB")
    data = corpus.test.to(exp.device)

    exp.model.eval()
    exp.predictor.eval()
    with torch.no_grad():

        for i in range(n_batches):
            print("Batch %d ---" % i)
            id_batch = data[batch_size * i : batch_size * (i + 1)]
            input_ = id_batch.reshape((batch_size, 1))
            print("IN", corpus.read_out(input_))
            phi = psi = x_b = None
            x_a_preds, x_b, phi, psi = exp.model(input_, x_b, phi, psi)
            predictor_outs = exp.predictor(x_b)
            max_probs = predictor_outs.argmax(dim=1)
            print("PRED", corpus.read_out(max_probs))


def parse_options():
    """parses the command line options for different settings."""
    optparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    optparser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=open,
        default="experiments.cfg",
        help="your experiments config file",
    )
    optparser.add_argument(
        "-n",
        "--num_cpus",
        dest="num_cpus",
        type=int,
        default=os.cpu_count() - 1,
        help="number of cpus you want to use",
    )
    optparser.add_argument(
        "-g",
        "--num_gpus",
        dest="num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="number of gpus you want to use",
    )
    optparser.add_argument(
        "-e",
        "--experiment",
        action="append",
        dest="experiments",
        help="run only selected experiments, by default run all experiments in "
        "config file.",
        required=True,
    )
    optparser.add_argument(
        "-C",
        "--checkpoint",
        dest="checkpoint",
        help="resume from checkpoint if found",
        required=True,
    )
    return optparser.parse_args()


if __name__ == "__main__":
    # Load and parse command line option and experiment configurations
    options = parse_options()
    configs = parse_config(options.config, options.experiments)

    # Run all experiments in parallel
    results = []
    for exp in configs:
        config = configs[exp]
        config["name"] = exp
        config["num_cpus"] = options.num_cpus
        config["num_gpus"] = options.num_gpus
        config["checkpoint"] = options.checkpoint

        path = os.path.expanduser(config.get("path", "~/nta/results"))
        config["path"] = path
        data_dir = os.path.expanduser(config.get("data_dir", "~/nta/datasets"))
        config["data_dir"] = data_dir
        config["checkpoint"] = os.path.expanduser(options.checkpoint)
        if "name" not in config:
            config["name"] = exp

        show_predictions(config)
