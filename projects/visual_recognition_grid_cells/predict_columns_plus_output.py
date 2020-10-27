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
from SDR_decoder import mlp_decoder


torch.manual_seed(18)
np.random.seed(18)

NUM_OBJECTS = 100
DATA_SET = 'mnist'

def predict_column_plus_reps(net, prediction_sequence, touch_sequence, label, numSensationsToInference, ground_truth):

    print("\n New prediction")
    plt.imsave('predicted_images/' + label + '_ground_truth.png', ground_truth)

    # for width_iter in range(5):
    #     for height_iter in range(5):
    for touch_iter in range(len(prediction_sequence)):

        # if touch_iter == 4:
        #     exit()


        print("Touch iter")
        print(touch_iter)

        # if touch_iter >= len(touch_sequence):
        #     # *** note touch sequence does not actually track how many touches were performed
        #     print("Completed all touches")
        #     break

        # touch_location = touch_sequence[touch_iter]
        # print("Touch location and sequence")
        # print(touch_location)
        # print(touch_sequence)

        input_SDR = np.zeros([128, 5*5])
        # print("Initial SDR")
        # print(np.shape(input_SDR))
        # print(input_SDR)

        # touch_indices = touch_sequence[0:touch_iter+1]
        # # need to convert these indices to those of the expected SDR, may be easier to flattten it first **
        # print("Touch indices")
        # print(touch_indices)
        # print(touch_indices[0])

        print("Prediction sequence")
        print(np.shape(prediction_sequence))
        # print("\n\n")
        # #print(prediction_sequence)
        # print("\n\n")
        current_sequence = prediction_sequence[touch_iter]
        # print(prediction_sequence[touch_iter])

        print("Current sequence")
        print(len(current_sequence))

        print(current_sequence)


        for sequence_iter in range(len(current_sequence)):
            if len(current_sequence[sequence_iter]) > 0:
                print("\nOn iter " + str(sequence_iter))
                print("Touch location")
                print(touch_sequence[sequence_iter])
                print("SDR indices")
                print(current_sequence[sequence_iter])
                # print("SDR before modification")
                # print(input_SDR[:, touch_sequence[sequence_iter]])
                input_SDR[current_sequence[sequence_iter], touch_sequence[sequence_iter]] = 1
                # print("SDR after modification")
                # print(input_SDR[:, touch_sequence[sequence_iter]])
            else:
                print("\nOn iter " + str(sequence_iter) + " of the sequence, there is no sensation or prediction to use")


        # print("Predicted SDR")
        # print(input_SDR)
        # print(np.shape(input_SDR))
        input_SDR = torch.from_numpy(np.reshape(input_SDR, 128*5*5))
        input_SDR = input_SDR.type(torch.DoubleTensor)
        # print(np.shape(input_SDR))
        # print(input_SDR.dtype)

        print("\nReconstructing:")
        reconstructed = net(input_SDR)
        # print(np.shape(reconstructed))

        # *** highlight predicted location - note that as going from a 5*5 space to a 28*28 space, this is only approximate
        current_touch = touch_sequence[touch_iter]
        print("Current touch")
        print(current_touch)
        print("Width and height iter")
        width_iter = current_touch//5
        height_iter = current_touch%5
        highlight_width_lower, highlight_width_upper = (1 + width_iter*5),  (1 + (width_iter+1)*5)
        highlight_height_lower, highlight_height_upper = (1 + height_iter*5),  (1 + (height_iter+1)*5)


        highlight_array = np.zeros((28,28))
        highlight_array[highlight_width_lower:highlight_width_upper, 
            highlight_height_lower:highlight_height_upper] = 0.5


        if numSensationsToInference != None:
            if touch_iter >= numSensationsToInference:
                #Add highlight to borders to indicate inference successful, and all 
                # future representations are based on model predictions
                highlight_array[0,:] = 1.0
                highlight_array[27,:] = 1.0
                highlight_array[:,0] = 1.0
                highlight_array[:,27] = 1.0


        # print("Highlight_array")
        # print(np.shape(highlight_array))
        # print(highlight_array)

        reconstructed = np.clip(reconstructed.detach().numpy() + highlight_array, 0, 1)


        # *** #random comparison - change the current 
        # print("Random control")
        # print(touch_sequence[touch_iter])
        # # random SDR (should have same sparsity as original SDRs)

        # controlSDR = input_SDR[touch_sequence[touch_iter]][np.nonzero(np.random())]
        # control_reconstructed = net(controlSDR)
        if numSensationsToInference != None:
            prediction = "correctly_classified"
        else:
            prediction = "misclassified"

        # print(np.shape(reconstructed[0,:,:]))
        # print(np.shape(ground_truth))

        plt.imsave('predicted_images/' + label + '_' + prediction + 
            '_touch_' + str(touch_iter) + '.png', reconstructed[0,:,:])

    #Save the final prediction without the prediction window 
    plt.imsave('predicted_images/' + label + '_' + prediction + 
        '_touch_' + str(touch_iter) + '.png', reconstructed[0,:,:])


        # plt.imsave('predicted_images/' + label + '_RandomControl_touch_' + str(touch_iter) + '.png')        


if __name__ == '__main__':

    if os.path.exists('predicted_images/') == False:
        try:
            os.mkdir('predicted_images/')
        except OSError:
            pass

    object_prediction_sequences = np.load('object_prediction_sequences.npy', allow_pickle=True, encoding='latin1')

    net = mlp_decoder().double()
    net.load_state_dict(torch.load('saved_networks/' + DATA_SET + '_decoder.pt'))

    for object_iter in range(NUM_OBJECTS):

        current_object = object_prediction_sequences[object_iter]

        # print(current_object)

        # for key in current_object.items():
        #     print(key)
        #     exit()

        predict_column_plus_reps(net, prediction_sequence=current_object['prediction_sequence'], 
            touch_sequence=current_object['touch_sequence'], label=current_object['name'], 
            numSensationsToInference=current_object['numSensationsToInference'],
            ground_truth=current_object['ground_truth_image'])
