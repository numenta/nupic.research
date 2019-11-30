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

# investigate how to load and save the dataset
# need to
# 1 - loop once through the dataset to apply the transforms
# 2 - save each batch of images
# 3 - write a custom dataset that load from these saved files instead
# https://discuss.pytorch.org/t/save-transformed-resized-images-after-dataloader/56464/6

# file to test run imagenet

import os
from time import time

from torch import utils, nn
from torchvision import models, transforms

# load the model
from nupic.research.frameworks.dynamic_sparse.common.datasets import CustomDataset
from nupic.research.frameworks.dynamic_sparse.models import BaseModel
from nupic.research.frameworks.pytorch.models.resnets import resnet50
from nupic.research.frameworks.pytorch.dataset_utils import CachedDatasetFolder

SMALL_IMAGENET = False

train_path = os.path.expanduser("~/nta/data/imagenet/train")
val_path = os.path.expanduser("~/nta/data/imagenet/val")
num_classes = 20

# preprocessing: https://github.com/pytorch/vision/issues/39
stats_mean = (0.485, 0.456, 0.406)
stats_std = (0.229, 0.224, 0.225)
train_transform = transforms.Compose(
    [
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(stats_mean, stats_std),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(stats_mean, stats_std),
    ]
)

# load train dataset
t0 = time()
train_dataset = CachedDatasetFolder(train_path, transform=train_transform, num_classes=num_classes)
print("Loaded train dataset")
t1 = time()
print("Time spent to load train dataset: {:.2f}".format(t1 - t0))

# load test dataset
t0 = time()
test_dataset = CachedDatasetFolder(val_path, transform=val_transform, num_classes=num_classes)
print("Loaded test dataset")
t1 = time()
print("Time spent to load test dataset: {:.2f}".format(t1 - t0))

# load dataloaders
t0 = time()
train_dataloader = utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
print("Loaded train dataloader")
test_dataloader = utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)
print("Loaded test dataloader")
t1 = time()
print("Time spent to load dataloaders: {:.2f}".format(t1 - t0))

# load network
t0 = time()
network = resnet50(config=dict(num_classes=num_classes))
# network = models.resnet50(pretrained=False)
# last_layer_shape = network.fc.weight.shape
# network.fc = nn.Linear(last_layer_shape[1], num_classes)

print("Loaded network")
t1 = time()
print("Time spent to load network: {:.2f}".format(t1 - t0))

# ------------------------- RUN MODEL

# simple base model
t0 = time()
exp_config = dict(device="cuda")
model = BaseModel(network, exp_config)
model.setup()
# simple dataset
dataset = CustomDataset(exp_config)
dataset.set_loaders(train_dataloader, test_dataloader)
epochs = 3
t1 = time()
print("Time spent to setup experiment: {:.2f}".format(t1 - t0))
for epoch in range(epochs):
    t0 = time()
    print("Running epoch {}".format(str(epoch)))
    log = model.run_epoch(dataset, epoch)
    t1 = time()
    print("Train acc: {:.4f}, Val acc: {:.4f}".format(log["train_acc"], log["val_acc"]))
    print("Time spent in epoch: {:.2f}".format(t1 - t0))
