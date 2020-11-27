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
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

torch.manual_seed(18)
np.random.seed(18)

data_set = 'mnist'
list_num_samples_per_class = [5]
WEIGHT_DECAY = 0.001
num_epochs = 1
knn_bool = True
knn_progressive_senations_bool = True
n_neighbors = 1

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

class RNN_model(torch.nn.Module):
    def __init__(self):
        super(RNN_model, self).__init__()

        self.hidden_size = 64
        self.num_layers = 1
        self.rnn = torch.nn.RNN(input_size=128, hidden_size=self.hidden_size, num_layers=self.num_layers,
            batch_first=True)
        self.fc = torch.nn.Linear(self.hidden_size, 10)

        print("Using an RNN classifier")

    def forward(self, x):

        # print("X shapes")
        # print(np.shape(x))

        hidden = torch.zeros(self.num_layers, np.shape(x)[0], self.hidden_size)

        x = x.reshape(-1, 5*5, 128)

        # print(np.shape(x)) 

        out, hidden = self.rnn(x, hidden)

        # print("Out shapes")
        # print(np.shape(out))

        out = out[:, -1, :] #Take the final representation

        # print(np.shape(out))

        # out = out.contiguous().view(-1, self.hidden_size)

        # print(np.shape(out))
        
        out = self.fc(out)

        # print(np.shape(out))

        return out


def sub_sample_classes(input_data, labels, num_samples_per_class, sanity_check=None):

    input_data_samples = []
    label_samples = []

    print("\n Loading " + str(num_samples_per_class) + " examples per class")

    if sanity_check == 'one_class_training':
        print("\nAs a sanity check, loading data for only a single class")
        num_classes = 1
    else:
        num_classes = 10

    for class_iter in range(num_classes):
        indices = np.nonzero(labels == class_iter)

        input_data_samples.extend(input_data[indices][0:num_samples_per_class])
        label_samples.extend(labels[indices][0:num_samples_per_class])

    print("Size of sub-samples")
    print(np.shape(input_data_samples))
    print(np.shape(label_samples))

    return input_data_samples, label_samples

def shuffle_SDR_order(input_data_samples, random_indices):

    SDR_shuffled_input_data_samples = []

    for image_iter in range(len(input_data_samples)):

        if arbitrary_SDR_order_bool == True:

            np.random.shuffle(random_indices)
            print("Shuffling the order of SDRs for each image")
        
        else: 
            print("Shuffling the order of SDRs using the same fixed sequence across images")

        # print(random_indices)

        # print("Original SDR:")
        # print(np.shape(input_data_samples[0]))
        temp_SDR_array = np.reshape(input_data_samples[image_iter], (128, 5*5))
        # print(np.shape(temp_SDR_array))
        # print(temp_SDR_array)
        random_SDR_array = temp_SDR_array[:, random_indices]
        SDR_shuffled_input_data_samples.append(np.reshape(random_SDR_array, (128*5*5)))

        # print("Shuffled SDR")
        # print(np.shape(SDR_shuffled_input_data_samples[0]))
        # print(SDR_shuffled_input_data_samples[0])
        

    print(np.shape(SDR_shuffled_input_data_samples))

    return SDR_shuffled_input_data_samples

def truncate_SDR_samples(input_data_samples, truncation_point):

    print("Truncating the number of sensations/SDR locations provided")

    truncated_input_data_samples = []

# len(input_data_samples)

    for image_iter in range(len(input_data_samples)):

        # print(np.shape(input_data_samples[0]))
        temp_SDR_array = np.reshape(input_data_samples[image_iter], (128, 5*5))
        # print(np.shape(temp_SDR_array))
        # print(temp_SDR_array)
        truncated_SDR_array = temp_SDR_array[:, 0:truncation_point+1]
        # print(np.shape(truncated_SDR_array))
        truncated_input_data_samples.append(np.reshape(truncated_SDR_array, (128*(truncation_point+1))))

        # print("Truncated SDR")
        # print(np.shape(truncated_input_data_samples[0]))
        # print(SDR_shuffled_input_data_samples[0])
        

    print(np.shape(truncated_input_data_samples))

    return truncated_input_data_samples

def load_data(data_section, random_indices, num_samples_per_class=5, sanity_check=None, data_set=data_set):

    input_data = np.load(data_set + '_SDRs_' + data_section + '.npy')
    labels = np.load(data_set + '_labels_' + data_section + '.npy')

    input_data_samples, label_samples = sub_sample_classes(input_data, labels, num_samples_per_class, sanity_check=None)

    input_data_samples = shuffle_SDR_order(input_data_samples, random_indices)

    return input_data_samples, label_samples

def kNN(n_neighbors, training_data, training_labels, testing_data, testing_labels):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    # knn.fit(np.reshape(training_data, (np.shape(training_data)[0], np.shape(training_data)[1]*np.shape(training_data)[2]*np.shape(training_data)[3])), training_labels)
    knn.fit(training_data, training_labels)

    acc = knn.score(testing_data, testing_labels)

    print("Accuracy of k-NN classifier " + str(acc))
    # print("Accuracy of k-NN classifier " + str(knn.score(np.reshape(testing_data, (np.shape(testing_data)[0], np.shape(testing_data)[1]*np.shape(testing_data)[2]*np.shape(testing_data)[3])), testing_labels)))

    return acc

def knn_progressive_senations(n_neighbors, training_data, training_labels, testing_data, testing_labels):

    acc_list = []

    for truncation_iter in range(25):

        truncated_training_data = truncate_SDR_samples(training_data, truncation_point=truncation_iter)
        truncated_testing_data = truncate_SDR_samples(testing_data, truncation_point=truncation_iter)

        acc_list.append(kNN(n_neighbors, truncated_training_data, training_labels, truncated_testing_data, testing_labels))

    print("All accuracies across truncation levels")
    print(acc_list)

    plt.scatter(list(range(1,26)), acc_list)
    plt.ylim(0,1)
    plt.show()


def train_net(net, training_data, training_labels, testing_data, testing_labels, lr):

    # input_data = torch.from_numpy(np.load(data_set + '_SDRs_' + data_section + '.npy'))
    # labels = torch.from_numpy(np.load(data_set + '_labels_' + data_section + '.npy'))

    training_data, training_labels, testing_data, testing_labels = (torch.FloatTensor(training_data), torch.LongTensor(training_labels),
        torch.FloatTensor(testing_data), torch.LongTensor(testing_labels))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=WEIGHT_DECAY)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        optimizer.zero_grad()

        # print(np.shape(training_data))
        # print(np.shape(training_labels))
        shuffle_indices = torch.randperm(len(training_labels))
        training_data = training_data[shuffle_indices,:]
        training_labels = training_labels[shuffle_indices]
        # print(training_labels[0:5])

        outputs = net(training_data)
        loss = criterion(outputs, training_labels)
        loss.backward()
        optimizer.step()

        training_acc = 100*(torch.sum(torch.argmax(outputs, dim=1)==training_labels)).item()/len(training_labels)
        print("\nEpoch:" + str(epoch))
        print("Training accuracy is " + str(training_acc))

        testing_acc = 100*(torch.sum(torch.argmax(net(testing_data), dim=1)==testing_labels)).item()/len(testing_labels)
        print("Testing accuracy is " + str(testing_acc))

    print('Finished Training')
    return testing_acc


def main_sim(num_samples_per_class):


    # Note the same fixed, random sampling of the input is used across training and testing, unless arbitrary_SDR_order_bool==True
    random_indices = np.arange(25)
    np.random.shuffle(random_indices)

    training_data, training_labels = load_data(data_section='training', random_indices=random_indices, num_samples_per_class=num_samples_per_class, sanity_check=None)

    print(np.shape(training_data))
    print(np.shape(training_labels))


    testing_data, testing_labels = load_data(data_section='testing', random_indices=random_indices, num_samples_per_class=100, sanity_check=None)

    print(np.shape(testing_data))
    print(np.shape(testing_labels))

    neighbour_parameters_lists = list(range(1, 11))

    rnn_lr_list = list(range(1, 11)) # for 1k epochs [0.005, 0.01, 0.05, 0.1, 0.5]

    acc_dic = {}

    if knn_bool == True:

        if knn_progressive_senations_bool == True:

            n_neighbors=1
            knn_progressive_senations(n_neighbors, training_data, training_labels, testing_data, testing_labels)

        else:

            for n_neighbors in neighbour_parameters_lists:

                acc_dic[str(n_neighbors)] = kNN(n_neighbors, training_data, training_labels, testing_data, testing_labels)

                with open('knn_parameter_resuts_' + str(num_samples_per_class) + '_samples_per_class.txt', 'w') as outfile:
                    json.dump(acc_dic, outfile)


    else:

        net = RNN_model()

        for lr in rnn_lr_list:

            acc_dic[str(lr)] = train_net(net, training_data, training_labels, testing_data, testing_labels, lr)
    
            with open('rnn_parameter_resuts_' + str(num_samples_per_class) + '_samples_per_class.txt', 'w') as outfile:
                json.dump(acc_dic, outfile)


if __name__ == '__main__':

    for num_samples_per_class in list_num_samples_per_class:

        main_sim(num_samples_per_class)
