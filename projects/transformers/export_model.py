# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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
Pretrained models need to be exported to be used for finetuning.
Only required argument for this script is the checkpoint folder.

Not tested for modified sparse models.
"""

import argparse
import os

from transformers import AutoModelForMaskedLM

# Import models. This will update Transformer's model mappings so that custom models can
# be loaded via AutoModelForMaskedLM.
import models # noqa F401


def save_pretrained(checkpoint_folder, destination_folder, model_name):
    if not model_name:
        model_name = os.path.split(checkpoint_folder)[-1]
    model = AutoModelForMaskedLM.from_pretrained(checkpoint_folder)
    destination_file_path = os.path.join(destination_folder, model_name)
    model.save_pretrained(destination_file_path)
    print(f"Model saved at {destination_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_folder", type=str,
                        help="Path to checkpoint to convert")
    parser.add_argument("--destination_folder", type=str,
                        default="/mnt/efs/results/pretrained-models/transformers-local",
                        help="Where to save the converted model")
    parser.add_argument("--model_name", type=str,
                        default=None,
                        help="Name of the model to save")
    args = parser.parse_args()
    save_pretrained(**args.__dict__)
