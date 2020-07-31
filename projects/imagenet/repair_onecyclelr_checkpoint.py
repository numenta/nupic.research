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
Remove bad parts from a checkpoint that make it dangerous to deserialize.
Workaround for https://github.com/pytorch/pytorch/issues/42376
"""

import argparse
import io
import pickle

from nupic.research.frameworks.pytorch.model_utils import (
    deserialize_state_dict,
    serialize_state_dict,
)


def repair(in_path):
    with open(in_path, "rb") as f:
        checkpoint = pickle.load(f)

    fix_needed = False
    if "lr_scheduler" in checkpoint:
        print("Loading LR scheduler state dict (this might take a few minutes)")
        with io.BytesIO(checkpoint["lr_scheduler"]) as buf:
            lr_sched_state_dict = deserialize_state_dict(buf)

        if "anneal_func" in lr_sched_state_dict:
            fix_needed = True
            del lr_sched_state_dict["anneal_func"]

            with io.BytesIO() as buf:
                serialize_state_dict(buf, lr_sched_state_dict)
                checkpoint["lr_scheduler"] = buf.getvalue()

            out_path = f"{in_path}.repaired"
            print(f"Saving {out_path}")
            with open(out_path, "wb") as f:
                pickle.dump(checkpoint, f)

    if not fix_needed:
        print("This checkpoint does not need repair")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="File path to checkpoint to repair")

    args = parser.parse_args()
    repair(args.checkpoint)
