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
This trains a simple CNN that can be used to output SDR features 
derived from images such as MNIST or Fashion-MNIST
'''
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from nupic.torch.modules import (
    KWinners2d, Flatten, rezero_weights, update_boost_strength
)
import torch

torch.manual_seed(18)
np.random.seed(18)

# Training parameters
TRAIN_NEW_NET = False
PERCENT_ON = 0.15
BOOST_STRENGTH = 20.0

CROSS_VAL = True # If true, use cross-validation subset of training data
CROSS_VAL_SPLIT = 0.1
DATASET = 'mnist'

LEARNING_RATE = 0.01
MOMENTUM = 0.5
EPOCHS = 10
FIRST_EPOCH_BATCH_SIZE = 4
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

def train(model, loader, optimizer, criterion, post_batch_callback=None):
    """
    Train the model using given dataset loader. 
    Called on every epoch.
    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: dataloader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
    :type optimizer: :class:`torch.optim.Optimizer`
    :param criterion: loss function to use
    :type criterion: function
    :param post_batch_callback: function(model) to call after every batch
    :type post_batch_callback: function
    """
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if post_batch_callback is not None:
            post_batch_callback(model)
        
def test(model, loader, test_size, criterion, epoch, test_data_bool):
    """
    Evaluate pre-trained model using given dataset loader.
    Called on every epoch.
    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: dataloader configured for the epoch.
    :type loader: :class:`torch.utils.data.DataLoader`
    :param criterion: loss function to use
    :type criterion: function
    :return: Dict with "accuracy", "loss" and "total_correct"
    """
    model.eval()
    loss = 0
    total_correct = 0

    #Store data for SDR-based classifiers
    all_SDRs = []
    all_labels = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            all_SDRs.append(np.array(model.output_sdr(data)))
            all_labels.append(target)

            loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            total_correct += pred.eq(target.view_as(pred)).sum().item()
    
        duty_cycle = (model.k_winner.duty_cycle[0,:,0,0]).numpy()
        print("\nMean duty cycle : " + str(np.mean(duty_cycle)))
        print("Stdev duty cycle: " + str(np.std(duty_cycle)))
        plt.hist(duty_cycle, bins=20, facecolor='crimson')
        plt.xlabel('Duty Cycle')
        plt.ylabel('Count')
        plt.xlim(0,0.6)
        plt.ylim(0,70)
        plt.show()
        # plt.savefig('duty_cycle_boost_' + str(BOOST_STRENGTH) + '_' + str(epoch) + '.png')
        # plt.clf()

        all_SDRs = np.concatenate(all_SDRs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

    if test_data_bool == True:
        print("Saving outputs from testing data")
        np.save(DATASET + '_SDRs_testing', all_SDRs)
        np.save(DATASET + '_labels_testing', all_labels)
    else:
        print("Saving outputs from training data")
        np.save(DATASET + '_SDRs_training', all_SDRs)
        np.save(DATASET + '_labels_training', all_labels)

    return {"accuracy": total_correct / test_size, 
            "loss": loss / test_size, 
            "total_correct": total_correct}

class SequentialSubSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


print("Creating model")
class sdr_cnn_base(nn.Module):
    def __init__(self, percent_on, boost_strength):
        super(sdr_cnn_base, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.k_winner = KWinners2d(channels=128, percent_on=percent_on, boost_strength=boost_strength, local=True)
        self.dense1 = nn.Linear(in_features=5*5*128, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=10)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.k_winner(x)
        x = x.view(-1, 5*5*128)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.softmax(self.output(x))

        return x

    def output_sdr(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.k_winner(x)
        print(np.shape(x))
        x = x.view(-1, 5*5*128)
        x = (x>0).float()

        return x

sdr_cnn = sdr_cnn_base(percent_on=PERCENT_ON, boost_strength=BOOST_STRENGTH)

sdr_cnn.to(device)

print("Loading data-sets")

if DATASET == 'mnist':
    normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=normalize)
    traing_len = len(train_dataset)

    if CROSS_VAL == True:
        print("Using hold-out cross-validation data-set for evaluating model")
        indices = range(traing_len)
        val_split = int(np.floor(CROSS_VAL_SPLIT*traing_len))
        train_idx, test_idx = indices[val_split:], indices[:val_split]
        # print(train_idx[0:10])
        # print(test_idx[0:10])
        # print(train_dataset.train_labels[train_idx][0:10])
        # print(train_dataset.train_labels[test_idx][0:10])
        # exit()
        test_dataset = datasets.MNIST('data', train=True, download=True, transform=normalize)

    elif CROSS_VAL == False:
        assert 1 == 0, "Not implemented"
        test_dataset = datasets.MNIST('data', train=False, transform=normalize)

    train_sampler = SequentialSubSampler(train_idx)
    test_sampler = SequentialSubSampler(test_idx)


# elif DATASET == 'fashion_mnist':
# *** need to fix normalizaiton ***
#     normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
#     train_dataset = datasets.FashionMNIST('data', train=True, download=True, transform=normalize)
#     test_dataset = datasets.FashionMNIST('data', train=False, transform=normalize)

# Configure data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, sampler=test_sampler)
first_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FIRST_EPOCH_BATCH_SIZE, sampler=train_sampler)


def post_batch(model):
    model.apply(rezero_weights)

if os.path.exists('saved_networks/') == False:
    try:
        os.mkdir('saved_networks/')
    except OSError:
        pass

if TRAIN_NEW_NET == True:

    print("Performing first epoch for update-boost-strength")
    sgd = optim.SGD(sdr_cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    train(model=sdr_cnn, loader=first_loader, optimizer=sgd, criterion=F.nll_loss, post_batch_callback=post_batch)

    sdr_cnn.apply(update_boost_strength)

    test(model=sdr_cnn, loader=test_loader, test_size=len(test_idx), epoch='pre_epoch', criterion=F.nll_loss, test_data_bool=False)

    print("Performing full training")
    for epoch in range(1, EPOCHS):
        train(model=sdr_cnn, loader=train_loader, optimizer=sgd, criterion=F.nll_loss, post_batch_callback=post_batch)
        sdr_cnn.apply(update_boost_strength)
        results = test(model=sdr_cnn, loader=test_loader, test_size=len(test_idx), epoch=epoch, criterion=F.nll_loss, test_data_bool=False)
        print(results)

    print("Saving network state...")
    torch.save(sdr_cnn.state_dict(), 'saved_networks/sdr_cnn.pt')

else:
    sdr_cnn.load_state_dict(torch.load('saved_networks/sdr_cnn.pt'))

    print("Evaluating a pre-trained model:")
    print("\nResults from training data-set")
    results = test(model=sdr_cnn, loader=train_loader, test_size=len(train_idx), epoch='final_train', criterion=F.nll_loss, test_data_bool=False) #Save SDRs from the training-data
    print(results)
    print("\nResults from testing data-set")
    results = test(model=sdr_cnn, loader=test_loader, test_size=len(test_idx), epoch='test', criterion=F.nll_loss, test_data_bool=True)
    print(results)
