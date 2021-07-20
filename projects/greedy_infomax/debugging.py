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

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision.datasets import STL10
from nupic.research.frameworks.greedy_infomax.models.FullModel import WrappedSuperGreedySmallSparseVisionModel
from nupic.research.frameworks.greedy_infomax.utils.data_utils import (
    supervised_dataset_args,
    unsupervised_dataset_args,
    validation_dataset_args,
)
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    module_specific_log_softmax_nll_loss,
    multiple_log_softmax_nll_loss,
)


def main():
    # load dataset
    dataset_args = unsupervised_dataset_args
    dataset_args.update(root="~/nta/data/STL10/")
    dataset = STL10(**dataset_args)
    subset = Subset(dataset, range(1000))
    dataloader = DataLoader(dataset=subset, batch_size=5, shuffle=False, num_workers=0)

    # create models of both kinds
    model = DataParallel(
        WrappedSuperGreedySmallSparseVisionModel(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
            sparsity=0.4,
        )
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    for (img, label) in dataloader:
        output = model.forward(img)
        encoding = model.module.encode(img)
        block_wise_losses = module_specific_log_softmax_nll_loss(output, None)
        loss = block_wise_losses[2]
        loss.backward()
        optimizer.step()
        break

    print("Done")


if __name__ == "__main__":
    main()
