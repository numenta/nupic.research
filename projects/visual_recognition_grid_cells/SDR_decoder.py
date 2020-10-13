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
train_net_bool = False
num_epochs = 10
batch_size = 64

if data_set == 'mnist':
    training_sources = datasets.MNIST('data', train=True, download=True).train_data.float()
    testing_sources = datasets.MNIST('data', train=False, download=True).test_data.float()
elif data_set == 'fashion_mnist':
    training_sources = datasets.FashionMNIST('data', train=True, download=True).train_data.float()
    testing_sources = datasets.FashionMNIST('data', train=False, download=True).test_data.float()

print(np.shape(testing_sources))
np.save("first_100_images", testing_sources[0:100])
exit()


class cnn_decoder(torch.nn.Module):
    def __init__(self):
        super(cnn_decoder, self).__init__()
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 5, stride=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 1, 5, stride=1)

    def forward(self, x):
        print(np.shape(x))
        x = x.view(-1, 5, 5, 64)
        x = self.unpool1(x)
        x = self.deconv1(x)
        x = self.unpool2(x)
        x = self.deconv1(x)
        print(np.shape(x))

        return x


class mlp_decoder(torch.nn.Module):
    def __init__(self):
        super(mlp_decoder, self).__init__()
        self.dense1 = nn.Linear(in_features=64*5*5, out_features=1024)
        self.dense2 = nn.Linear(in_features=1024, out_features=28*28)

    def forward(self, x):
        #print(np.shape(x))
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        #print(np.shape(x))
        x = x.view(-1, 28, 28)
        #print(np.shape(x))

        return x

def initialize():

    net = mlp_decoder()

    training_input = torch.from_numpy(np.load(data_set + '_SDRs_training.npy'))
    testing_input = torch.from_numpy(np.load(data_set + '_SDRs_testing.npy'))

    #Note the 'sources' are the original image that needs to be reconstructed

    if data_set == 'mnist':
        training_sources = datasets.MNIST('data', train=True, download=True).train_data.float()
        testing_sources = datasets.MNIST('data', train=False, download=True).test_data.float()
    elif data_set == 'fashion_mnist':
        training_sources = datasets.FashionMNIST('data', train=True, download=True).train_data.float()
        testing_sources = datasets.FashionMNIST('data', train=False, download=True).test_data.float()


    #Labels are loaded from the npy data and saved along with the source image as a sanity check 
    # that they correctly correspond
    training_labels = torch.from_numpy(np.load(data_set + '_labels_training.npy'))
    testing_labels = torch.from_numpy(np.load(data_set + '_labels_testing.npy'))

    return net, training_input, testing_input, training_sources, testing_sources, training_labels, testing_labels

def train_net(net, training_input, testing_input, training_sources, testing_sources, training_labels, testing_labels):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0

        for batch_iter in range(math.ceil(len(training_labels)/batch_size)):
            batch_input = training_input[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]
            batch_sources = training_sources[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]

            # print(np.shape(batch_input))
            # print("\nBatch sources:")
            # print(np.shape(batch_sources))

            optimizer.zero_grad()

            reconstructed = net(batch_input)
            # print(np.shape(reconstructed))
            # print(reconstructed.dtype)
            # print(batch_sources.dtype)
            loss = criterion(reconstructed, batch_sources)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("\nEpoch:" + str(epoch))
        print("Training loss is " + str(running_loss/10000))

    print("Saving network state...")
    torch.save(net.state_dict(), 'saved_networks/' + data_set + '_decoder.pt')


    print('Finished Training')

def evaluate_decoder(net, training_input, testing_input, training_sources, testing_sources, training_labels, testing_labels):

    net.load_state_dict(torch.load('saved_networks/' + data_set + '_decoder.pt'))

    for batch_iter in range(1):
        batch_input = training_input[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]
        batch_sources = training_sources[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]
        batch_labels = training_labels[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_labels))]

        reconstructed = net(batch_input)

        for image_iter in range(len(batch_labels)):

            plt.imsave('output_images/' + str(batch_iter) + '_' + str(image_iter) + '_original_label:' 
                + str(batch_labels[image_iter].item()) + '.png', batch_sources.detach().numpy()[image_iter])
            plt.imsave('output_images/' + str(batch_iter) + '_' + str(image_iter) + '_reconstructed_label:' 
                + str(batch_labels[image_iter].item()) + '.png', reconstructed.detach().numpy()[image_iter])

if __name__ == '__main__':

    if os.path.exists('output_images/') == False:
        try:
            os.mkdir('output_images/')
        except OSError:
            pass

    net, training_input, testing_input, training_sources, testing_sources, training_labels, testing_labels = initialize()

    if train_net_bool == True:
        print("Training new network")
        train_net(net, training_input, testing_input, training_sources, testing_sources, training_labels, testing_labels)
    
    elif train_net_bool == False:
        print("Evaluating previously trained network")
        evaluate_decoder(net, training_input, testing_input, training_sources, testing_sources, training_labels, testing_labels)

