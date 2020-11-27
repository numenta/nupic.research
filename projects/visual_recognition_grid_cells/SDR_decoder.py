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

torch.manual_seed(18)
np.random.seed(18)

DATASET = 'mnist'
TRAIN_NEW_NET = False
EPOCHS = 2
BATCH_SIZE = 64

class mlp_decoder(torch.nn.Module):
    def __init__(self):
        super(mlp_decoder, self).__init__()
        self.dense1 = nn.Linear(in_features=128*5*5, out_features=512)
        self.dense2 = nn.Linear(in_features=512, out_features=28*28)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = torch.sigmoid(self.dense2(x))
        x = x.view(-1, 28, 28)

        return x

def initialize():

    net = mlp_decoder()

    # SDR inputs that the decoder needs to use to reconstruct images
    training_input = torch.from_numpy(np.load(DATASET + '_SDRs_base_net_training.npy'))
    testing_input = torch.from_numpy(np.load(DATASET + '_SDRs_SDR_classifiers_training.npy'))

    # The 'sources' are the original images that need to be reconstructed
    if DATASET == 'mnist':
        print("Using MNIST data-set")
        total_sources = datasets.MNIST('data', train=True, download=True).train_data.float()/255
        testing_SDR_classifiers_dataset = datasets.MNIST('data', train=False, download=True).train_data.float()/255 # Not used
        # by auto-encoder but used to save output images from Torchvision for later use by GridCellNet

    elif DATASET == 'fashion_mnist':
        print("Using Fashion-MNIST data-set")
        total_sources = datasets.FashionMNIST('data', train=True, download=True).train_data.float()/255
        testing_SDR_classifiers_dataset = datasets.FashionMNIST('data', train=False, download=True).train_data.float()/255

    total_len = len(total_sources)

    print("Using hold-out cross-validation data-set for evaluating model")
    indices = range(total_len) 
    val_split = int(np.floor(0.1*total_len))
    train_idx, test_decoder_idx = indices[val_split:], indices[:val_split]
    
    training_sources = total_sources[train_idx]
    testing_decoder_sources = total_sources[test_decoder_idx]

    training_labels = torch.from_numpy(np.load(DATASET + '_labels_base_net_training.npy'))
    testing_labels = torch.from_numpy(np.load(DATASET + '_labels_SDR_classifiers_training.npy'))

    np.save(DATASET + "_images_SDR_classifiers_training", testing_decoder_sources)
    np.save(DATASET + "_images_SDR_classifiers_testing", testing_SDR_classifiers_dataset)

    return net, training_input, testing_input, training_sources, testing_decoder_sources, training_labels, testing_labels

def train_net(net, training_input, training_sources, training_labels):

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3) #SGD(net.parameters(), lr=0.1, momentum=0.9)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times

        running_loss = 0

        for batch_iter in range(math.ceil(len(training_labels)/BATCH_SIZE)):
            batch_input = training_input[batch_iter*BATCH_SIZE:min((batch_iter+1)*BATCH_SIZE, len(training_labels))]
            batch_sources = training_sources[batch_iter*BATCH_SIZE:min((batch_iter+1)*BATCH_SIZE, len(training_labels))]
            batch_labels = training_labels[batch_iter*BATCH_SIZE:min((batch_iter+1)*BATCH_SIZE, len(training_labels))]

            optimizer.zero_grad()
            reconstructed = net(batch_input)
            loss = criterion(reconstructed, batch_sources)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print("\nEpoch:" + str(epoch))
        print("Training loss is " + str(running_loss/len(training_labels)))

    print("Saving network state...")
    torch.save(net.state_dict(), 'saved_networks/' + DATASET + '_decoder.pt')

    print('Finished Training')

def generate_images(net, net_input, sources, labels):

    net.load_state_dict(torch.load('saved_networks/' + DATASET + '_decoder.pt'))

    # Re-construct one batch worth of testing examples and save as images
    for batch_iter in range(1):
        batch_input = net_input[batch_iter*BATCH_SIZE:min((batch_iter+1)*BATCH_SIZE, len(labels))]
        batch_sources = sources[batch_iter*BATCH_SIZE:min((batch_iter+1)*BATCH_SIZE, len(labels))]
        batch_labels = labels[batch_iter*BATCH_SIZE:min((batch_iter+1)*BATCH_SIZE, len(labels))]

        reconstructed = net(batch_input)

        for image_iter in range(len(batch_labels)):

            plt.imsave('output_images/' + str(batch_iter) + '_' + str(image_iter) + '_original_label_' 
                + str(batch_labels[image_iter].item()) + '.png', batch_sources.detach().numpy()[image_iter])
            plt.imsave('output_images/' + str(batch_iter) + '_' + str(image_iter) + '_reconstructed_label_' 
                + str(batch_labels[image_iter].item()) + '.png', reconstructed.detach().numpy()[image_iter])


if __name__ == '__main__':

    if os.path.exists('output_images/') == False:
        try:
            os.mkdir('output_images/')
        except OSError:
            pass

    net, training_input, testing_input, training_sources, testing_sources, training_labels, testing_labels = initialize()

    if TRAIN_NEW_NET == True:
        print("Training new network")
        train_net(net, training_input, training_sources, training_labels)
        print("Generating images from newly trained network using unseen data")
        generate_images(net, net_input=testing_input, sources=testing_sources, labels=testing_labels)
    
    elif TRAIN_NEW_NET == False:
        print("Generating images from previously trained network using unseen data")
        generate_images(net, net_input=testing_input, sources=testing_sources, labels=testing_labels)
        #evaluate_decoder(net, net_input=testing_input, sources=testing_sources, labels=testing_labels)


