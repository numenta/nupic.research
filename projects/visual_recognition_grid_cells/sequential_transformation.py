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
Transform image-data (e.g. MNIST or Fashion-MNIST) into sequences (random or ordered)
of local pixel patches
'''

import numpy as np

data_set = 'mnist'
shuffle_patches_bool = False
num_samples_per_class = 5
sanity_check = None


def convert_to_sequences(image_input, patch_size=7):


    return None

def shuffle_sequences():

    return None

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

    print(np.shape(input_data_samples))
    print(np.shape(label_samples))

    return None

def load_data(data_section='training', num_samples_per_class, sanity_check=None):

    input_data = torch.from_numpy(np.load(data_set + '_SDRs_' + data_section + '.npy'))
    labels = torch.from_numpy(np.load(data_set + '_labels_' + data_section + '.npy'))

    return None


if __name__ == '__main__':
    
    limited_training_data = 
    limited_training_labels = 




    