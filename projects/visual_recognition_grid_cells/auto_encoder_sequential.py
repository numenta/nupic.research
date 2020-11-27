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
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

from nupic.torch.modules import (
    KWinners, KWinners2d, Flatten, rezero_weights, update_boost_strength
)

torch.manual_seed(18)
np.random.seed(18)

data_set = 'mnist'
train_net_bool = True
randomize_order_bool = False
generate_SDR_patches_bool = False
sample_patches_bool = False
noise_level = 0.3
CROSS_VAL = True # If true, use cross-validation subset of training data
CROSS_VAL_SPLIT = 0.1
# LEARNING_RATE = 0.001
# MOMENTUM = 0.5

num_epochs = 3
batch_size = 128
percent_on = 0.15
boost_strength = 10.0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device: " + str(device))

class mlp_auto_encoder(torch.nn.Module):
    def __init__(self):
        super(mlp_auto_encoder, self).__init__()
        self.dense1_encode = nn.Linear(in_features=7*7, out_features=256)
        self.dense2_encode = nn.Linear(in_features=256, out_features=128)
        # self.k_winner = KWinners(n=128, percent_on=percent_on, boost_strength=boost_strength)
        self.k_winner = KWinners2d(channels=128, percent_on=percent_on, boost_strength=boost_strength, local=True)
        self.dense1_decode = nn.Linear(in_features=128, out_features=256)
        self.dense2_decode = nn.Linear(in_features=256, out_features=7*7)

    def encode(self, x):
        #print(np.shape(x))
        x = x.reshape(-1, 7*7)
        x = F.relu(self.dense1_encode(x))
        x = F.relu(self.dense2_encode(x))
        # print("\n\n New forward pass")
        # print(x[0])

        # '2D' based k-winner

        # print(x[0])
        # print(np.shape(x))
        x = x.reshape(-1, 128, 1, 1)
        # print(np.shape(x))
        # 1-D based k-winner
        x = self.k_winner(x)
        x = x.reshape(-1, 128)
        # print("\nK-winner x")
        # print(x[0])


        return x

    def decode(self, x):
        #print(np.shape(x))
        x = F.relu(self.dense1_decode(x))
        x = torch.sigmoid(self.dense2_decode(x))
        #print(np.shape(x))
        x = x.view(-1, 7, 7)
        #print(np.shape(x))

        return x
    
    def forward(self, x):
        #print(np.shape(x))
        x = self.encode(x)
        x = self.decode(x)
        #print(np.shape(x))

        return x

    def extract_SDRs(self, x):

        x = self.encode(x)
        indices = x>0
        x[indices] = 1

        return x

# class hinton_auto_encoder(mlp_auto_encoder):
#     def __init__(self):
#         super(hinton_auto_encoder, self).__init__()
#         self.dense1_encode = nn.Linear(in_features=28*28, out_features=256)
#         self.dense2_encode = nn.Linear(in_features=256, out_features=256)
#         self.dense3_encode = nn.Linear(in_features=256, out_features=128)
#         self.dense1_decode = nn.Linear(in_features=128, out_features=256)
#         self.dense2_decode = nn.Linear(in_features=256, out_features=256)
#         self.dense3_decode = nn.Linear(in_features=256, out_features=28*28)


#     def encode(self, x):
#         #print(np.shape(x))
#         x = x.reshape(-1, 28*28)
#         x = F.relu(self.dense1_encode(x))
#         x = F.relu(self.dense2_encode(x))
#         x = torch.sigmoid(self.dense3_encode(x))

#         if self.training == True:
#             x = x + torch.randn(size=x.shape)*16

#         return x

#     def decode(self, x):

#         x = F.relu(self.dense1_decode(x))
#         x = F.relu(self.dense2_decode(x))
#         x = torch.sigmoid(self.dense3_decode(x))
#         x = x.view(-1, 28, 28)

#         return x


class half_cnn_auto_encoder_twenty_eight(torch.nn.Module):
    def __init__(self):
        super(half_cnn_auto_encoder_twenty_eight, self).__init__()
        self.dropout_noise = nn.Dropout(p = 0.1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.k_winner = KWinners2d(channels=128, percent_on=percent_on, boost_strength=boost_strength, local=True)


        self.dense_decode1 = nn.Linear(in_features=128*5*5, out_features=256)
        self.dense_decode2 = nn.Linear(in_features=256, out_features=28*28)

        print("\nUsing a CNN-encoding component")

        print("\n\n***Applying dropout to inputs!")

    def encode(self, x):

        # print("\n New encode pass")
        # print(np.shape(x))
        x = x.reshape(-1, 1, 28, 28) # Add channel dimension

        # print(x[0])
        
        # if self.training == True:
        #     # add noise to some examples
        #     # if np.random.uniform() > 0.5:
        #         # print("Adding noise")
        #     x = torch.clamp(x + torch.FloatTensor(np.random.normal(0, scale=noise_level, size=x.shape)), 0, 1)
        #     # else:
        #     #     print("Not adding noise")
        # # print(x[0])

        x = self.dropout_noise(x)

        # print(np.shape(x))
        # print(np.shape(x))
        # print("Conv1:")
        x = F.relu(self.conv1(x))
        # print(np.shape(x))
        x = self.pool1(x)
        # print(np.shape(x))
        # print("Conv2:")
        x = F.relu(self.conv2(x))
        # print(np.shape(x))
        x = self.pool2(x)
        # print(np.shape(x))

        x = self.k_winner(x)
        # print(np.shape(x))

        return x

    def decode(self, x):

        x = x.view(-1, 5*5*128)
        x = F.relu(self.dense_decode1(x))
        x = torch.sigmoid(self.dense_decode2(x))
        x = x.view(-1, 28, 28)

        return x

    def forward(self, x):

        x = self.encode(x)
        x = self.decode(x)

        return x

    def extract_SDRs(self, x):

        x = self.encode(x)
        # indices = x>0
        # x[indices] = 1
        x = x.view(-1, 5*5*128)
        x = (x>0).float()
        print("Extracted SDR")
        print(x[0])

        return x


# class half_cnn_auto_encoder(mlp_auto_encoder):
#     def __init__(self):
#         super(half_cnn_auto_encoder, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, padding=1)
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, padding=0)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.k_winner = KWinners2d(channels=128, percent_on=percent_on, boost_strength=boost_strength, local=True)

#         print("Using a CNN-encoding component")

#     def encode(self, x):

#         # print("\n New encode pass")

#         x = x.reshape(-1, 1, 7, 7) # Add channel dimension
#         # print(np.shape(x))
#         # print("Conv1:")
#         x = F.relu(self.conv1(x))
#         # print(np.shape(x))
#         x = self.pool1(x)
#         # print(np.shape(x))
#         # print("Conv2:")
#         x = F.relu(self.conv2(x))
#         # print(np.shape(x))
#         x = self.pool2(x)
#         # print(np.shape(x))

#         x = x.reshape(-1, 128, 1, 1)

#         x = self.k_winner(x)


#         x = x.reshape(-1, 128)

#         return x



def sample_patches(input_images):

    print(np.shape(input_images))

    assembled_patches = []

    # Iterate over images
    for image_iter in range(len(input_images)):

        # Iterate over patches in the image
        for patch_width_iter in range(4):

            for patch_height_iter in range(4):
                
                assembled_patches.append(input_images.detach().numpy()[image_iter][patch_width_iter*7:(patch_width_iter+1)*7, 
                    patch_height_iter*7:(patch_height_iter+1)*7])

    print(np.shape(assembled_patches))

    return assembled_patches

def initialize(sample_patches_bool):

    net = half_cnn_auto_encoder_twenty_eight()

    #Note the 'sources' are the original image that needs to be reconstructed

    print("\nNot normalizing data\n")
    normalize = None

    if data_set == 'mnist':

        print("\nUsing MNIST data-set")
        training_sources = datasets.MNIST('data', train=True, download=True, transform=normalize).train_data.float()/255
        training_labels = datasets.MNIST('data', train=True, download=True, transform=normalize).train_labels

    elif data_set == 'fashion_mnist':

        print("\nUsing Fashion-MNIST data-set")
        training_sources = datasets.FashionMNIST('data', train=True, download=True, transform=normalize).train_data.float()/255

    traing_len = len(training_sources)

    if CROSS_VAL == True:
        print("\nUsing hold-out cross-validation data-set for evaluating model")
        indices = range(traing_len) 
        val_split = int(np.floor(CROSS_VAL_SPLIT*traing_len))
        train_idx, test_idx = indices[val_split:], indices[:val_split]
        training_sources = training_sources[train_idx]
        training_labels = training_labels[train_idx]

        if data_set == 'mnist':

            testing_sources = datasets.MNIST('data', train=True, download=True, transform=normalize).train_data.float()[test_idx]/255
            testing_labels = datasets.MNIST('data', train=True, download=True, transform=normalize).train_labels[test_idx]

        elif data_set == 'fashion_mnist':

            testing_sources = datasets.FashionMNIST('data', train=True, download=True, transform=normalize).train_data.float()[test_idx]/255


    if sample_patches_bool == True:
        training_sources = torch.FloatTensor(sample_patches(training_sources))
        testing_sources = torch.FloatTensor(sample_patches(testing_sources))

    return net, training_sources, testing_sources, training_labels, testing_labels

def train_net(net, training_sources):

    criterion = torch.nn.MSELoss()
    #optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    net.train()

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0

        for batch_iter in range(math.ceil(len(training_sources)/batch_size)):

            batch_sources = training_sources[batch_iter*batch_size:min((batch_iter+1)*batch_size, len(training_sources))]

            optimizer.zero_grad()

            reconstructed = net(batch_sources)

            loss = criterion(reconstructed, batch_sources)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        net.apply(update_boost_strength)

        duty_cycle = (net.k_winner.duty_cycle).numpy()
        print(np.shape(duty_cycle))
        duty_cycle = (net.k_winner.duty_cycle[0,:,0,0]).numpy()
        print(np.shape(duty_cycle))
        print("\nMean duty cycle : " + str(np.mean(duty_cycle)))
        print("Stdev duty cycle: " + str(np.std(duty_cycle)))
        plt.hist(duty_cycle, bins=20, facecolor='crimson')
        plt.xlabel('Duty Cycle')
        plt.ylabel('Count')
        plt.xlim(0,0.6)
        plt.ylim(0,70)
        # plt.show()
        plt.savefig('duty_cycle_boost_' + str(boost_strength) + '_' + str(epoch) + '.png')
        plt.clf()

        print("\nEpoch:" + str(epoch))
        print("Training loss is " + str(running_loss/len(training_sources)))

    print("Saving network state...")
    torch.save(net.state_dict(), 'saved_networks/' + data_set + '_patch_autoencoder.pt')

    print('Finished Training')

def evaluate_auto_encoder(net, source):

    net.load_state_dict(torch.load('saved_networks/' + data_set + '_patch_autoencoder.pt'))

    net.eval()

    print("Outputing a sample of the trained auto-encodrs predictions")
    num_example_patches = 10

    reconstructed = net(source[0:num_example_patches])

    for image_iter in range(num_example_patches):

        plt.imsave('output_images/' + str(image_iter) + '_original_patch.png', source.detach().numpy()[image_iter])
        plt.imsave('output_images/' + str(image_iter) + '_reconstructed_patch.png', reconstructed.detach().numpy()[image_iter])

def generate_full_size_SDRs(net, input_images, all_labels, test_data_bool=False, randomize_order_bool=False):

    net.load_state_dict(torch.load('saved_networks/' + data_set + '_patch_autoencoder.pt'))

    net.eval()

    num_image_examples = 1000 # len(input_images)
    print("Generating SDRs for " + str(num_image_examples) + " images")

    all_image_SDRs = net.extract_SDRs(input_images[0:num_image_examples]).detach().numpy()
    all_labels_output = all_labels.detach().numpy()[0:num_image_examples]

    print("\nShape of outputs")
    print(np.shape(all_image_SDRs))
    print(np.shape(all_labels_output))

    # all_image_SDRs = np.concatenate(all_image_SDRs, axis=0)
    # all_labels_output = np.concatenate(all_labels_output, axis=0)


    if test_data_bool == True:
        print("Saving outputs from testing data")
        np.save(data_set + '_SDRs_testing', all_image_SDRs)
        np.save(data_set + '_labels_testing', all_labels_output)
        np.save(data_set + "_images_testing", input_images.detach().numpy())

    else:
        print("Saving outputs from training data")
        np.save(data_set + '_SDRs_training', all_image_SDRs)
        np.save(data_set + '_labels_training', all_labels_output)
        np.save(data_set + "_images_training", input_images.detach().numpy())



def generate_patch_wise_SDRs(net, input_images, all_labels, test_data_bool=False, randomize_order_bool=False):

    net.load_state_dict(torch.load('saved_networks/' + data_set + '_patch_autoencoder.pt'))

    net.eval()

    num_image_examples = 200 # len(input_images)

    print("Generating SDRs for " + str(num_image_examples) + " images")

    all_image_SDRs = []
    all_labels_output = []

    for image_iter in range(num_image_examples):

        image_SDRs = np.zeros((4,4,128))
        image_reconstruct = np.zeros((28,28))

        random_width_locations = list(range(4))
        random.shuffle(random_width_locations)
        # print(random_width_locations)
        random_height_locations = list(range(4))
        random.shuffle(random_height_locations)
        # print(random_height_locations)

        for patch_width_iter in range(4):

            for patch_height_iter in range(4):

                input_patch = input_images[image_iter][patch_width_iter*7:(patch_width_iter+1)*7, 
                    patch_height_iter*7:(patch_height_iter+1)*7]

                input_patch = input_patch[None, :, :]

                # print("\n*Input* patch")
                # print(np.shape(input_patch))
                
                # print("Net outputs")
                # print(np.shape(net.encode(input_patch).detach().numpy()))
                # print(np.shape(net(input_patch)))
                # print(net.encode(input_patch).detach().numpy())
                # print(net(input_patch))

                # Note image reconstruction is never randomized
                image_reconstruct[patch_width_iter*7:(patch_width_iter+1)*7, 
                    patch_height_iter*7:(patch_height_iter+1)*7] = net(input_patch).detach().numpy()

                if randomize_order_bool == True:
                    patch_width_index = random_width_locations[patch_width_iter]
                    patch_height_index = random_height_locations[patch_height_iter]

                else:
                    patch_width_index = patch_width_iter
                    patch_height_index = patch_height_iter

                # print("Width and heigh indices:")
                # print(patch_width_index)
                # print(patch_height_index)

                image_SDRs[patch_width_index, patch_height_index, :] = net.extract_SDRs(input_patch).detach().numpy()

                # print("Image SDRs and reconstruct patch")
                # print(np.shape(image_SDRs))
                # print(np.shape(image_reconstruct))
                # print(image_SDRs)
                # print(image_reconstruct)

        current_label = all_labels[image_iter].detach().numpy()
        all_labels_output.append(current_label)
        all_image_SDRs.append(image_SDRs)

        # plt.imsave('output_images/' + str(image_iter) + '_original_image_label_' + str(current_label) + '.png', input_images.detach().numpy()[image_iter])
        # plt.imsave('output_images/' + str(image_iter) + '_reconstructed_image_label_' + str(current_label) + '.png', image_reconstruct)

    print(np.shape(all_image_SDRs))
    print(np.shape(all_labels_output))

    # all_image_SDRs = np.concatenate(all_image_SDRs, axis=0)
    # all_labels_output = np.concatenate(all_labels_output, axis=0)


    if test_data_bool == True:
        print("Saving outputs from testing data")
        np.save(data_set + '_SDRs_testing', all_image_SDRs)
        np.save(data_set + '_labels_testing', all_labels_output)
        np.save(data_set + "_images_testing", input_images.detach().numpy())

    else:
        print("Saving outputs from training data")
        np.save(data_set + '_SDRs_training', all_image_SDRs)
        np.save(data_set + '_labels_training', all_labels_output)
        np.save(data_set + "_images_training", input_images.detach().numpy())


if __name__ == '__main__':

    if os.path.exists('output_images/') == False:
        try:
            os.mkdir('output_images/')
        except OSError:
            pass
  
    if os.path.exists('saved_networks/') == False:
        try:
            os.mkdir('saved_networks/')
        except OSError:
            pass

    net, training_sources, testing_sources, training_labels, testing_labels = initialize(sample_patches_bool)

    if train_net_bool == True:
        print("Training new network")
        train_net(net, training_sources)
        print("Evaluating newly trained network")
        evaluate_auto_encoder(net, source=testing_sources)
    
    elif train_net_bool == False:
        print("Evaluating previously trained network")
        evaluate_auto_encoder(net, source=testing_sources)

    if generate_SDR_patches_bool == True:

        net, training_sources, testing_sources, training_labels, testing_labels = initialize(sample_patches_bool=False)

        print("Generating patch-based SDRs for training data")
        generate_patch_wise_SDRs(net, input_images=training_sources, all_labels=training_labels, test_data_bool=False, randomize_order_bool=randomize_order_bool)

        print("Generating patch-based SDRs for testing data")
        generate_patch_wise_SDRs(net, input_images=testing_sources, all_labels=testing_labels, test_data_bool=True, randomize_order_bool=randomize_order_bool)

    else:

        print("Generating full-size SDRs for training data")
        generate_full_size_SDRs(net, input_images=training_sources, all_labels=training_labels, test_data_bool=False, randomize_order_bool=randomize_order_bool)

        print("Generating full-size SDRs for testing data")
        generate_full_size_SDRs(net, input_images=testing_sources, all_labels=testing_labels, test_data_bool=True, randomize_order_bool=randomize_order_bool)
