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
Calculate how many sensations were required 
'''

import numpy as np
import os
import math
import matplotlib.pyplot as plt



DATA_SET = 'mnist'

def plot_sensations_to_inference(object_prediction_sequences):

    num_sensations_list = []
    correctly_classified_list = []

    # *** need to deal with "None" when it wasn't inferred ***


    for object_iter in range(len(object_prediction_sequences)):

        print("\n\nNew object:")
        print(object_prediction_sequences[object_iter]['numSensationsToInference'])
        print(object_prediction_sequences[object_iter]['correctly_classified'])

        num_sensations_list.append(object_prediction_sequences[object_iter]['numSensationsToInference'])
        correctly_classified_list.append(object_prediction_sequences[object_iter]['correctly_classified'])

    print(np.shape(num_sensations_list))
    print(num_sensations_list[0:5])

    cumm_percent_inferred = []
    num_correct = 0

    print("Accuracy based on sensations == None")
    print(np.nonzero(np.array(num_sensations_list)!=None))
    print(np.shape(np.nonzero(np.array(num_sensations_list)!=None)))
    print(len(np.nonzero(np.array(num_sensations_list)!=None)[0]))
    print(len(np.nonzero(np.array(num_sensations_list)!=None)[0])/len(object_prediction_sequences))

    print("Accuracy based correctly classified bool")
    print(np.nonzero(np.array(correctly_classified_list)==True))
    print(np.shape(np.nonzero(np.array(correctly_classified_list)==True)))
    print(np.sum(np.nonzero(np.array(correctly_classified_list)==True))/len(object_prediction_sequences))

    print("Number of objects " + str(len(object_prediction_sequences)))

    for num_sensation_iter in range(25):

        num_correct += len(np.nonzero(np.array(num_sensations_list)==(num_sensation_iter+1))[0])
    
        cumm_percent_inferred.append(num_correct/len(object_prediction_sequences))


    plt.scatter(list(range(1,26)), cumm_percent_inferred)
    plt.ylim(0,1)
    plt.show()

    print(cumm_percent_inferred)


if __name__ == '__main__':

    if os.path.exists('predicted_images/') == False:
        try:
            os.mkdir('predicted_images/')
        except OSError:
            pass

    object_prediction_sequences = np.load('object_prediction_sequences.npy', allow_pickle=True, encoding='latin1')


    plot_sensations_to_inference(object_prediction_sequences)    
    

