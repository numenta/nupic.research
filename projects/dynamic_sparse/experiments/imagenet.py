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
# ----------------

# file to test run imagenet

from torchvision import models, datasets, transforms
from torch import nn, utils
import os

# cifar10 stats
# stats_mean = (0.4914, 0.4822, 0.4465)
# stats_std = (0.2023, 0.1994, 0.2010)
# imagenet stats
stats_mean = (0.485, 0.456, 0.406)
stats_std = (0.229, 0.224, 0.225)
transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(stats_mean, stats_std)])

# dataset = datasets.CIFAR10(root=os.path.expanduser("~/nta/datasets"),
#     transform=transform, download=True)
dataset = datasets.ImageNet(root=os.path.expanduser("~/nta/data/imagenet"), 
    transform=transform)
data_loader = utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

model = models.resnet50()
# required to remove head if smaller dataset
# last_layer_shape = model.fc.weight.shape
# model.fc = nn.Linear(last_layer_shape[1], 10)
loss_func = nn.CrossEntropyLoss()

epochs = 1
for epoch in range(epochs):
    for x,y in data_loader:
        y_pred = model(x)
        # import pdb;pdb.set_trace()
        loss = loss_func(y_pred, y)
        print("Loss: {:.4f}".format(loss))

