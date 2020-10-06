#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
#
'''
Trains a decoder to reconstruct input images from SDRs
'''

import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

import inspect

data_set = 'mnist'
train_net_bool = True
num_epochs = 50
batch_size = 64

torch.manual_seed(18)
np.random.seed(18)

TRAIN_BATCH_SIZE = 64
normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
# train_dataset = datasets.MNIST('data', train=True, download=True, transform=normalize)
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

# for batch_idx, (data, target) in enumerate(train_loader):
#     print(batch_idx)
#     break

#training_sources = datasets.MNIST('data', train=True, download=True).train_data.float()
#true_labels = datasets.MNIST('data', train=True, download=True).train_labels
#testing_sources = datasets.MNIST('data', train=False, download=True).test_data.float()

train_dataset = datasets.MNIST('data', train=True, download=True, transform=normalize)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


#Labels are loaded from the npy data and saved along with the source image as a sanity check 
# that they correctly correspond
training_labels = torch.from_numpy(np.load(data_set + '_labels_training.npy'))
#testing_labels = torch.from_numpy(np.load(data_set + '_labels_testing.npy'))


for batch_iter, (data, true_targets) in enumerate(train_loader):
    print(batch_iter)
    print(np.shape(data))
    print(np.shape(true_targets))

    batch_labels = training_labels[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]
    #true_batch_labels = true_labels[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]

    print(true_targets[0:10])
    print(batch_labels[0:10])

    break


# for batch_iter in range(math.ceil(len(training_labels)/batch_size)):
#     batch_labels = training_labels[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]
#     true_batch_labels = true_labels[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]

#     print(true_batch_labels[0:10])
#     print(batch_labels[0:10])

#     break


