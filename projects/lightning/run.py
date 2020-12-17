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

import inspect
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch

from experiments import CONFIGS


def main(args):
    config = CONFIGS[args.experiment]
    lightning_model_class = config["lightning_model_class"]
    lightning_model_args = config.get("lightning_model_args", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lightning_model_class(**lightning_model_args).to(device)

    trainer_args = config.get("lightning_trainer_args", {})

    # Use command line args to overwrite args in the dict
    # TODO: distinguish between user-specified and defaaults. (currently
    # unspecified args can still overwrite the ones in the dict.)
    valid_trainer_args = inspect.signature(pl.Trainer.__init__).parameters
    trainer_args.update({k: v
                         for k, v in vars(args).items()
                         if k in valid_trainer_args})

    if hasattr(lightning_model_class, "trainer_requirements"):
        trainer_args.update(lightning_model_class.trainer_requirements)

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment", default="default",
                        help="Experiment to run", choices=CONFIGS.keys())
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
