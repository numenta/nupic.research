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
This trains basic classifiers on SDRs derived from images
'''

import numpy as np
import logging.config
import torch
import torch.nn.functional as F

from sklearn.neighbors import KNeighborsClassifier

data_set = 'fashion_mnist'
shuffle_SDRs_bool = True

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(5*5*128, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)
        print("\nUsing an MLP classifier")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class linear(torch.nn.Module):
    def __init__(self):
        super(linear, self).__init__()
        self.fc = torch.nn.Linear(5*5*128, 10)
        print("\nUsing a linear classifier")

    def forward(self, x):
        # x = x.view(-1, 5*5, 64)
        # x = torch.transpose(x)
        # print(np.shape(x))

        # x = x[:, 2, 2, :]
        # print(np.shape(x))
        x = self.fc(x)
        return x

def train_net():

    net = MLP()

    training_data = torch.from_numpy(np.load(data_set + '_SDRs_training.npy'))
    training_labels = torch.from_numpy(np.load(data_set + '_labels_training.npy'))

    # if shuffle_SDRs_bool == True:
    #     for input_iter in len(training_labels:
    #         training_data[input_iter, :, :]

    limited_samples = 50
    print("\nOnly providing " + str(limited_samples) + " for training!")
    training_data = training_data[0:limited_samples]
    training_labels = training_labels[0:limited_samples]
    print(np.shape(training_labels))
    print(np.shape(training_data))


    testing_data = torch.from_numpy(np.load(data_set + '_SDRs_testing.npy'))
    testing_labels = torch.from_numpy(np.load(data_set + '_labels_testing.npy'))



    knn = KNeighborsClassifier(n_neighbors=1)

    knn.fit(training_data, training_labels)

    print("Accuracy of knn " + str(knn.score(testing_data, testing_labels)))

    # criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=1.0, momentum=0.9)

    # for epoch in range(1):  # loop over the dataset multiple times

    #     optimizer.zero_grad()

    #     outputs = net(training_data)
    #     loss = criterion(outputs, training_labels)
    #     loss.backward()
    #     optimizer.step()

    #     training_acc = 100*(torch.sum(torch.argmax(outputs, dim=1)==training_labels)).item()/len(training_labels)
    #     print("\nEpoch:" + str(epoch))
    #     print("Training accuracy is " + str(training_acc))

    #     testing_acc = 100*(torch.sum(torch.argmax(net(testing_data), dim=1)==testing_labels)).item()/len(testing_labels)
    #     print("Testing accuracy is " + str(testing_acc))

    # print('Finished Training')

if __name__ == '__main__':
    train_net()
    