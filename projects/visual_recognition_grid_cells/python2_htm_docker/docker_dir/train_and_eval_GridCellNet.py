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
Train and evaluate a network based on grid-cell path integration
to perform visual object recognition given a sequence of SDR feature representations
Much of this code is based on the doExperiment function under capacity_simulation.py in the 
location_in_the_neocortex_a_theory_of_sensorimotor_object_recognition_using_cortical_grid_cells
repository.
'''

import math
import os
import random
import time
import json

import numpy as np

from GridCellNet_base import (
  PIUNCorticalColumnForVisualRecognition, PIUNExperimentForVisualRecognition)
from pre_process_SDRs import generate_image_objects

# Core parameters for model urnning
THRESHOLD_LIST = [0.7] # The different dendritic thresholds to try using for the GridCellNet classifier
# Note this determines the threshold for the connections from feature columns to grid cells 
# Adjusting the dendritic threshold for grid cells to feature columns appears less important for performance on generalization
# Recommend values in the range of 0.6-0.8 for learning with 5 examples per class
NUM_SAMPLES_PER_CLASS_LIST = [5] # The number of examples per class to give during training, e.g. [1, 5, 10, 20],
# with each one run as a separate model. Note using number of examples per class of 100+ requires hours of training time
NUM_TEST_OBJECTS_PER_CLASS = 10 # The number of test objects per class to use during evaluation
NUM_TRIALS = 1 # The number of network duplicates to run for estimating performance distributions
DATASET = 'mnist' # Options are 'mnist' or 'fashion_mnist', although functionality for fashion-mnist may be incomplete
SEED1=1
SEED2=2

# Setting up special conditions for model learning and evaluation
FIXED_TOUCH_SEQUENCE = None; #Use a random but fixed sequence when performing inference for every object; options are
# None or range(5*5) --> use the latter if a fixed sequence is desired
if FIXED_TOUCH_SEQUENCE != None:
  random.shuffle(FIXED_TOUCH_SEQUENCE)
EVAL_ON_TRAINING_DATA_BOOL=False # If True, assess recall of objects learned during training rather than
# generalization to new objects; note due to how targets are created, this only operates as expected 
# if NUM_SAMPLES_PER_CLASS_LIST and NUM_TEST_OBJECTS_PER_CLASS both equal [1]
SANITY_CHECK = None # Options are None or 'one_class_training', the latter which assesses performance when the
# training data only contains examples from one class

def object_learning_and_inference(EVAL_ON_TRAINING_DATA_BOOL,
                 locationModuleWidth,
                 feature_columns_to_grid_cells_threshold,
                 grid_cells_to_feature_columns_threshold,
                 bumpType,
                 cellCoordinateOffsets,
                 cellsPerColumn,
                 activeColumnCount,
                 columnCount,
                 featuresPerObject,
                 objectWidth,
                 numModules,
                 seed1,
                 seed2,
                 anchoringMethod):

  if seed1 != None:
    print("Setting random seed")
    np.random.seed(seed1)

  if seed2 != None:
    print("Setting random seed")
    random.seed(seed2)

  locationConfigs = []

  perModRange = float((90.0 if bumpType == "square" else 60.0) /
                      float(numModules))
  
  print("\nUsing L4 dendritic thresholds of " + str(int(math.ceil(feature_columns_to_grid_cells_threshold*activeColumnCount))))
  print("Using L6 dendritic thresholds of " + str(int(math.ceil(grid_cells_to_feature_columns_threshold*numModules))) + "\n")

  for i in xrange(numModules):
    orientation = (float(i) * perModRange) + (perModRange / 2.0)

    # Note these parameters are for the L6 layer (grid cells), some of which are reused (below) for the *L4* layer
    config = {
      "cellsPerAxis": locationModuleWidth,
      "scale": 40.0,
      "orientation": np.radians(orientation),
      "activationThreshold": int(math.ceil(feature_columns_to_grid_cells_threshold*activeColumnCount)), # Note difference in threshold to L4 layer
      "initialPermanence": 1.0,
      "connectedPermanence": 0.5,
      "learningThreshold": int(math.ceil(feature_columns_to_grid_cells_threshold*activeColumnCount)), 
      "sampleSize": -1, # during learning, setting this to -1 means max new synapses = len(activeInput)
      "permanenceIncrement": 0.1,
      "permanenceDecrement": 0.0,
      "cellCoordinateOffsets": cellCoordinateOffsets,
      "anchoringMethod": anchoringMethod
    }

    locationConfigs.append(config)

  l4Overrides = {
    "initialPermanence": 1.0,
    "activationThreshold": int(math.ceil(grid_cells_to_feature_columns_threshold*numModules)),
    "reducedBasalThreshold": int(math.ceil(grid_cells_to_feature_columns_threshold*numModules)),
    "minThreshold": numModules,
    "sampleSize": numModules,
    "cellsPerColumn": cellsPerColumn,
    "columnCount": columnCount
  } 

  allLocationsAreUnique = None 

  print("Using " + str(num_samples_per_class) + " training objects per class")

  column = PIUNCorticalColumnForVisualRecognition(locationConfigs, L4Overrides=l4Overrides,
                              bumpType=bumpType)

  train_features_dic, train_objects, object_images = generate_image_objects(DATASET, num_samples_per_class, objectWidth, 
      locationModuleWidth, data_set_section='SDR_classifiers_training', sanity_check=SANITY_CHECK)
  
  ColumnPlusNet = PIUNExperimentForVisualRecognition(column, features_dic=train_features_dic)

  if EVAL_ON_TRAINING_DATA_BOOL == True:
    print("Evaluating recall of objects seen during training")
    test_features_dic, test_objects, object_images = train_features_dic, train_objects, object_images

  elif EVAL_ON_TRAINING_DATA_BOOL == False:
    print("Using novel (never seen during training) objects to evaluate accuracy")

    test_features_dic, test_objects, object_images = generate_image_objects(DATASET, NUM_TEST_OBJECTS_PER_CLASS, objectWidth,
                            locationModuleWidth, data_set_section='SDR_classifiers_testing')

  currentLocsUnique = True

  print("\nLearning objects")
  for objectDescription in train_objects:
    print("Learning " + objectDescription["name"])
    start_learn = time.time()

    objLocsUnique = ColumnPlusNet.learnObject(objectDescription)
    currentLocsUnique = currentLocsUnique and objLocsUnique

    print("Time to learn object : " + str(time.time() - start_learn))

  numFailures = 0 # General failures on classificaiton, including either converged to an incorrect label, or never converged
  numIncorrect = 0 
  numNeverConverged = 0
  total_sensations = 0

  # Over-write the features dictionary with those that are being used for evaluation
  ColumnPlusNet.features = test_features_dic

  print("\nGetting class and non-class targets")
  allClassTargets = []

  for class_iter in range(10):
    print("Targets of " + str(class_iter))
    start_get_targets = time.time()
    allClassTargets.append(ColumnPlusNet.getClassFeatures(objectClass=str(class_iter), featuresPerObject=featuresPerObject))
  
  object_iter = 0
  object_prediction_sequences = []

  for objectDescription in test_objects:
    
    start_infer = time.time()
    numSensationsToInference, incorrect, prediction_sequence, touch_sequence = ColumnPlusNet.inferObjectWithRandomMovements(
        objectDescription,
        objectImage=object_images[object_iter],
        allClassTargets=allClassTargets,
        cellsPerColumn=cellsPerColumn,
        trial_iter=ii,
        fixed_touch_sequence=FIXED_TOUCH_SEQUENCE)
    
    # object_prediction_sequences is used by downstream programs to e.g. visualise the predictions of the network
    object_prediction_sequences.append({'name':objectDescription["name"],
      'prediction_sequence':prediction_sequence,
      'touch_sequence':touch_sequence,
      'correctly_classified':numSensationsToInference!=None,
      'numSensationsToInference':numSensationsToInference,
      'ground_truth_image':object_images[object_iter]})

    object_iter += 1
    if numSensationsToInference is None:
      numFailures += 1
      numNeverConverged += incorrect['never_converged']
      numIncorrect += incorrect['false_convergence']
    else:
      print("numSensationsToInference:")
      print(numSensationsToInference)
      total_sensations += numSensationsToInference #Keep a running tally of sensations needed on 
      # successful trials to calculate the mean number of sensations needed for inference
    print("Time to infer object : " + str(time.time() - start_infer))

  print("Saving prediction sequences...")
  np.save('prediction_data/object_prediction_sequences', object_prediction_sequences, allow_pickle=True)

  mean_sensations = None

  print("Number of test objects: " + str(NUM_TEST_OBJECTS_PER_CLASS*10))
  accuracy = 100* (NUM_TEST_OBJECTS_PER_CLASS*10 - numFailures) / float(NUM_TEST_OBJECTS_PER_CLASS*10)
  errors = 100 * numFailures / float(NUM_TEST_OBJECTS_PER_CLASS*10)
  false_converging = 100 * numIncorrect / float(NUM_TEST_OBJECTS_PER_CLASS*10)
  never_converged = 100 * numNeverConverged / float(NUM_TEST_OBJECTS_PER_CLASS*10)
  if (NUM_TEST_OBJECTS_PER_CLASS*10 - numFailures) > 0:
    mean_sensations = total_sensations / float(NUM_TEST_OBJECTS_PER_CLASS*10 - numFailures) #Divide by number of successful sensations
  allLocationsAreUnique = currentLocsUnique

  result = {
    "num_samples_per_class": num_samples_per_class,
    "num_test_objects_per_class" : NUM_TEST_OBJECTS_PER_CLASS,
    "accuracy" : accuracy,
    "total_errors" : errors,
    "false_converging": false_converging,
    "never_converged" : never_converged,
    "mean_sensations" : mean_sensations,
    "allLocationsAreUnique": allLocationsAreUnique,
  }

  print result
  return result


if __name__ == '__main__':

  if os.path.exists('misclassified/') == False:
      try:
          os.mkdir('misclassified/')
      except OSError:
          pass

  if os.path.exists('correctly_classified/') == False:
      try:
          os.mkdir('correctly_classified/')
      except OSError:
          pass

  if os.path.exists('results/') == False:
      try:
          os.mkdir('results/')
      except OSError:
          pass

  if os.path.exists('prediction_data/') == False:
      try:
          os.mkdir('prediction_data/')
      except OSError:
          pass

  numOffsets = 2 # Spreads out the activation of grid cells to model uncertainty about location
  cellCoordinateOffsets = tuple([i * (0.998 / (numOffsets-1)) + 0.001
                                 for i in xrange(numOffsets)])
  for threshold_iter in range(len(THRESHOLD_LIST)):
    
    feature_columns_to_grid_cells_threshold = THRESHOLD_LIST[threshold_iter]

    for training_objects_iter in range(len(NUM_SAMPLES_PER_CLASS_LIST)):

      num_samples_per_class = NUM_SAMPLES_PER_CLASS_LIST[training_objects_iter]

      all_results = {}

      for ii in range(NUM_TRIALS):

        all_results["trial_" + str(ii)] = object_learning_and_inference(
                         EVAL_ON_TRAINING_DATA_BOOL=EVAL_ON_TRAINING_DATA_BOOL,
                         locationModuleWidth=50, # This is the main parameter to consider increasing for 
                         # challenging generalization tasks (e.g. learning many objects)
                         feature_columns_to_grid_cells_threshold=feature_columns_to_grid_cells_threshold, #dendritic threshold for connections from the 
                         # cortical columns representing input features, synapsing on the grid cells
                         grid_cells_to_feature_columns_threshold=0.9,
                         bumpType='square', # No significance of using this or vs e.g. Gaussian bump for 
                         # this task
                         cellsPerColumn=32,
                         columnCount=128, # Total dimension of the input feature SDRs
                         activeColumnCount=19, # Number of non-zero values in the input feature SDRs
                         cellCoordinateOffsets=cellCoordinateOffsets,
                         featuresPerObject=5*5,
                         objectWidth=5,
                         numModules=40, # 2nd parameter to increase (after locationModuleWidth) for 
                         # greater representational capacity
                         seed1=SEED1,
                         seed2=SEED2,
                         anchoringMethod="corners") # May see a performance improvement with 
                         # 'discrete', which assumes a noise-less setting; not yet tried

        print("Trial results:")
        print(all_results)

        with open('results/GridCellNet_threshold_' + str(feature_columns_to_grid_cells_threshold) + '_num_samples_per_class_' + str(num_samples_per_class) + '_all_results.json', 'w') as outfile:
            json.dump(all_results, outfile)
