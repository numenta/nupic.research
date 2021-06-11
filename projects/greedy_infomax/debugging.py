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

from nupic.research.frameworks.greedy_infomax.models import SparseFullVisionModel


def main():
    # create models of both kinds
    vernon_model = SparseFullVisionModel(
            negative_samples=16,
            block_dims = [3, 4, 6],
            num_channels = [64, 128, 128],
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
    )

    total_params = sum([x.view(-1).shape[0] for x in vernon_model.parameters()])
    required_sparsity  = 3310336./total_params
    print("Done")


if __name__ == "__main__":
    main()
