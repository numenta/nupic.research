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


import torch

from projects.greedy_infomax.experiments import CONFIGS
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import (
    all_module_multiple_log_softmax,
)
from nupic.research.frameworks.greedy_infomax.Greedy_InfoMax.GreedyInfoMax.vision\
    .models.FullModel import FullVisionModel
from optparse import OptionParser
import time


if __name__ == "__main__":
    experiment_config = CONFIGS["full_resnet_50"]
    experiment_config["distributed"] = False
    experiment_config["batch_size"]=1
    experiment_config["lr_scheduler_class"] = None
    experiment_class = experiment_config["experiment_class"]()
    experiment_class.setup_experiment(experiment_config)

    model_1 = experiment_class.encoder
    opt_1 = experiment_class.encoder_optimizer

    # load parameters and options
    parser = OptionParser()
    (opt, _) = parser.parse_args()
    opt.negative_samples = 16
    opt.resnet = 50
    opt.grayscale = 1
    opt.model_splits = 3
    opt.loss = 0
    opt.weight_init = False
    opt.prediction_step = 5
    opt.train_module = 3
    opt.device = torch.device("cpu")

    model_2 = FullVisionModel(opt, calc_loss=True)
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        print(f"OUR MODEL:{p1.mean()}, {p1.var()}")
        print(f"SINDY'S MODEL:{p2.mean()}, {p2.var()}")
        p2.data = p1.data.clone()

    opt_2 = torch.optim.SGD(model_2.parameters(), lr=2e-4)

    data_u = experiment_class.unsupervised_loader
    data_s = experiment_class.supervised_loader
    data_v = experiment_class.val_loader

    model_2.switch_calc_loss(True)
    print_idx = 100
    starttime = time.time()
    cur_train_module = opt.train_module

    for step, (img, label) in enumerate(data_u):
        starttime = time.time()
        model_input = img.to(opt.device)
        label = label.to(opt.device)

        #calculate forward for vernon model
        torch.manual_seed(1)
        out_1 = model_1(model_input)
        loss_1 = all_module_multiple_log_softmax(out_1, None).sum()

        torch.manual_seed(1)
        #calculate loss forward for gim mod
        loss_2, _, _, accuracy_2 = model_2(model_input, label, n=cur_train_module)
        loss_2 = loss_2.sum()  # Take mean over outputs of different GPUs.
        accuracy_2 = torch.mean(accuracy_2, 0)


        loss_1.backward()
        loss_2.backward()

        for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
            assert torch.all(p1.grad == p2.grad)

        opt_1.step()
        opt_2.step()
        break

    print("EQUAL")