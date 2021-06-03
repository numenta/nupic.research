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

import threading

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision.datasets import STL10

from nupic.research.frameworks.greedy_infomax.Greedy_InfoMax.GreedyInfoMax.vision.arg_parser import (
    arg_parser,
)
from nupic.research.frameworks.greedy_infomax.Greedy_InfoMax.GreedyInfoMax.vision.models import (
    load_vision_model,
)
from nupic.research.frameworks.greedy_infomax.models import FullVisionModel
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
    dataset = STL10(**unsupervised_dataset_args)
    dataset_supervised = STL10(**supervised_dataset_args)
    dataset_validation = STL10(**validation_dataset_args)
    subset = Subset(dataset, range(1000))
    dataloader = DataLoader(dataset=subset, batch_size=5, shuffle=False, num_workers=0)

    # create models of both kinds
    vernon_model = DataParallel(
        FullVisionModel(
            negative_samples=16,
            k_predictions=5,
            resnet_50=False,
            grayscale=True,
            patch_size=16,
            overlap=2,
        )
    )

    vernon_optimizer = torch.optim.Adam(vernon_model.parameters(), lr=2e-4)

    opt = arg_parser.parse_args()
    arg_parser.create_log_path(opt)
    opt.training_dataset = "unlabeled"

    # random seeds
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # load model
    gim_model, gim_optimizer = load_vision_model.load_model_and_optimizer(opt)

    gim_model_dict = gim_model.state_dict()
    vernon_model_dict = vernon_model.state_dict()

    vernon_model.load_state_dict(
        {k: v for k, v in zip(vernon_model_dict.keys(), gim_model_dict.values())}
    )

    for (img, label) in dataloader:
        vernon_output = vernon_model.forward(img)
        vernon_encoding = vernon_model.module.encode(img)
        vernon_loss = module_specific_log_softmax_nll_loss(vernon_output, None)
        gim_loss, gim_c, gim_h, gim_accuracy = gim_model.forward(img, None)
        total_gim_loss = gim_loss.sum()
        total_vernon_loss = vernon_loss.sum()
        total_vernon_loss.backward()
        total_gim_loss.backward()
        vernon_optimizer.step()
        gim_optimizer.step()
        break

    print("Done")


if __name__ == "__main__":
    main()
