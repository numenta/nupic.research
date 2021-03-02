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

"""
Code to create column-networks with grid-cell representations that can
perform visual object recognition requiring generalization to unseen objects
"""

import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from htmresearch.algorithms.apical_tiebreak_temporal_memory import (
    ApicalTiebreakPairMemory,
)
from htmresearch.algorithms.location_modules import Superficial2DLocationModule
from htmresearch.frameworks.location.path_integration_union_narrowing import (
    PIUNCorticalColumn,
    PIUNExperiment,
)


class PIUNCorticalColumnForVisualRecognition(PIUNCorticalColumn):
    """
    A L4 + L6a network. Sensory input causes minicolumns in L4 to activate,
    which drives activity in L6a. Motor input causes L6a to perform path
    integration, updating its activity, which then depolarizes cells in L4.

    Whenever the sensor moves, call movementCompute. Whenever a sensory input
    arrives, call sensoryCompute.

    Adds method that provides an easily accessible list of the location
    representation across modules for copying, and over-writes initializaiton to
    correctly specify column cell dimensions.

    """
    def __init__(self, locationConfigs,  # noqa: N803
                 L4Overrides=None, bumpType="gaussian"):
        """
        @param L4Overrides (dict)
        Custom parameters for L4

        @param locationConfigs (sequence of dicts)
        Parameters for the location modules
        """
        self.bumpType = bumpType

        l4_cell_count = L4Overrides["columnCount"] * L4Overrides["cellsPerColumn"]

        if bumpType == "square":
            self.L6aModules = [
                Superficial2DLocationModule(
                    anchorInputSize=l4_cell_count,
                    **config)
                for config in locationConfigs]
        else:
            raise ValueError("Invalid bumpType", bumpType)

        l4_params = {
            "columnCount": 128,  # Note overriding below
            "cellsPerColumn": 16,
            "basalInputSize": sum(module.numberOfCells()
                                  for module in self.L6aModules)
        }

        if L4Overrides is not None:
            l4_params.update(L4Overrides)

        self.L4 = ApicalTiebreakPairMemory(**l4_params)

    def get_location_copy(self):
        """
        Get the population representation of the location layer for copying.
        """
        active_cells_list = []

        for module in self.L6aModules:

            active_cells_list.append(module.getActiveCells())

        return active_cells_list


class PIUNExperimentForVisualRecognition(PIUNExperiment):
    """
    An experiment class which passes sensory and motor inputs into a special two
    layer network and tracks the location of a sensor on an object.

    This version enables target classes to be used that evaluate generalization to
    unseen examples, as well as inputs derived from images such as MNIST.

    """

    def __init__(self, column,
                 features_dic=None,
                 noiseFactor=0,  # noqa: N803
                 moduleNoiseFactor=0,
                 num_grid_cells=40*50*50,
                 num_classes=10):
        """
        @param column (PIUNColumn)
        A two-layer network.

        @param featureNames (list)
        A list of the features that will ever occur in an object.

        Overwrite the original initializer to enable loading image-based features
        """
        self.column = column

        # Weights to learn associations between active locations and class labels
        self.class_weights = np.zeros((num_grid_cells, num_classes))

        # Use these for classifying SDRs and for testing whether they're correct.
        # Allow storing multiple representations, in case the experiment learns
        # multiple points on a single feature. (We could switch to indexing these by
        # objectName, featureIndex, coordinates.)
        # Example:
        # (objectName, featureIndex): [(0, 26, 54, 77, 101, ...), ...]
        self.locationRepresentations = defaultdict(list)
        self.inputRepresentations = {
            # Example:
            # (objectName, featureIndex, featureName): [0, 26, 54, 77, 101, ...]
        }

        # Load the set of features from the image-based data
        self.features = features_dic

        # For example:
        # [{"name": "Object 1",
        #   "features": [
        #       {"top": 40, "left": 40, "width": 10, "height" 10, "name": "A"},
        #       {"top": 80, "left": 80, "width": 10, "height" 10, "name": "B"}]]
        self.learnedObjects = []

        # The location of the sensor. For example: {"top": 20, "left": 20}
        self.locationOnObject = None

        self.maxSettlingTime = 10
        self.maxTraversals = 1

        self.monitors = {}
        self.nextMonitorToken = 1

        self.noiseFactor = noiseFactor
        self.moduleNoiseFactor = moduleNoiseFactor

        self.representationSet = set()

    def getClassFeatures(self,  # noqa: N802
                         objectClass,  # noqa: N803
                         featuresPerObject):
        """
        For all the feature locations, get all of the location representations for a
        given class (e.g. all examples of "9"s in MNIST, as opposed to particular
        examples of a 9)

        @param objectClass (str)
        The target class, e.g. "2"

        @param featuresPerObject (int)

        @return class_targets (list)
        Target representations for the given class
        """
        class_targets = []

        for i_feature in range(featuresPerObject):

            name_iter = 0
            target_representations_temp = np.array([])

            while len(self.locationRepresentations[(objectClass
                                                   + "_" + str(name_iter), 0)]) > 0:

                target_representations_temp = np.concatenate(
                    (target_representations_temp, np.concatenate(
                        self.locationRepresentations[
                            (objectClass + "_" + str(name_iter), i_feature)])))

                name_iter += 1

            class_targets.append(set(target_representations_temp))

        return class_targets

    def learnObject(self,
                  objectDescription,
                  randomLocation=False,
                  useNoise=False,
                  noisyTrainingTime=1):
        """
        Train the network to recognize the specified object. Move the sensor to one of
        its features and activate a random location representation in the location
        layer. Move the sensor over the object, updating the location representation
        through path integration. At each point on the object, form reciprocal
        connections between the represention of the location and the representation
        of the sensory input.
        @param objectDescription (dict)
        For example:
        {"name": "Object 1",
         "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                      {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"}]}
        @return locationsAreUnique (bool)
        True if this object was assigned a unique set of locations. False if a
        location on this object has the same location representation as another
        location somewhere else.
        """
        self.reset()
        self.column.activateRandomLocation()

        locationsAreUnique = True
        all_locations = []

        if randomLocation or useNoise:
            numIters = noisyTrainingTime
        else:
            numIters = 1
        for i in xrange(numIters):
            for iFeature, feature in enumerate(objectDescription["features"]):
                self._move(feature, randomLocation=randomLocation, useNoise=useNoise)
                featureSDR = self.features[feature["name"]]
                self._sense(featureSDR, learn=True, waitForSettle=False)

                locationRepresentation = self.column.getSensoryAssociatedLocationRepresentation()
                self.locationRepresentations[(objectDescription["name"],
                                              iFeature)].append(locationRepresentation)
                self.inputRepresentations[(objectDescription["name"],
                                           iFeature, feature["name"])] = (
                                             self.column.L4.getWinnerCells())

                locationTuple = tuple(locationRepresentation)
                locationsAreUnique = (locationsAreUnique and
                                      locationTuple not in self.representationSet)

                # Track all the grid cells active over learning
                all_locations.extend(locationRepresentation)
                print("All locations")
                print(np.shape(all_locations))

                self.representationSet.add(tuple(locationRepresentation))

            # Update the weights associating location reps with class labels
            unique_locations = list(set(all_locations))
            print("\nActual max location cell:" + str(np.max(unique_locations)))
            print("Theoretical max location cell:" + str(np.shape(self.class_weights)[0]))
            #print(unique_locations)
            print("\nUnique locations:")
            print(np.shape(unique_locations))
            print(np.shape(np.array(unique_locations)))

            print("\nClass weights")
            print(np.shape(self.class_weights))
            print(np.shape(self.class_weights[unique_locations, int(objectDescription["name"][0])]))

            #print(self.class_weights)
            # Index by active grid-cells and the true class (i.e. supervised signal)
            self.class_weights[unique_locations, int(objectDescription["name"][0])] += 1
            print("Num non-zero values")
            print(np.sum(self.class_weights))

            #int(objectDescription["name"][0])

            #print(self.class_weights)


        self.learnedObjects.append(objectDescription)

        return locationsAreUnique


    def inferObjectWithRandomMovements(self,  # noqa: C901, N802
                                         objectDescription,  # noqa: N803
                                         objectImage,  # noqa: N803
                                         all_class_targets,
                                         cellsPerColumn,  # noqa: N803
                                         trial_iter,
                                         class_threshold,
                                         fixed_touch_sequence=None,
                                         numSensations=None,
                                         randomLocation=False):
        """
        Attempt to recognize the specified object with the network. Moves
        the sensor over the object until the object is recognized.

        @param objectDescription (dict)
        For example:
        {"name": "Object 1",
         "features": [{"top": 0, "left": 0, "width": 10, "height": 10, "name": "A"},
                      {"top": 0, "left": 10, "width": 10, "height": 10, "name": "B"}]}

        @objectImage (Numpy array)
        The current object's image

        @all_class_targets (list)
        Target representations for each class

        @cellsPerColumn (int)

        @trial_iter (int)

        @param numSensations (int or None)
        Set this to run the network for a fixed number of sensations. Otherwise this
        method will run until the object is recognized or until maxTraversals is
        reached.

        @return inferredStep (int or None), incorrect (dic), prediction_sequence (list),
        touchSequence (list)
        """
        self.reset()

        for monitor in self.monitors.values():
            monitor.beforeInferObject(objectDescription)

        currentStep = 0  # noqa: N806
        finished = False
        inferred = False
        inferredStep = None  # noqa: N806
        prevTouchSequence = None  # noqa: N806
        incorrect = {"never_converged": 1, "false_convergence": 0}  # Track if the
        # non-recognition was due to convergance to an incorrect representation or
        # never converging

        for _ in xrange(self.maxTraversals):  # noqa: F821
            # Choose touch sequence.
            while True:
                touchSequence = range(len(objectDescription["features"]))  # noqa: N806
                if fixed_touch_sequence is None:
                    random.shuffle(touchSequence)
                    print("\nPerforming inference using an arbitrary, unfixed"
                          "sequense of touches:")
                    print(touchSequence)

                else:
                    print("\nPerforming inference using a fixed random"
                          "sequense of touches:")
                    touchSequence = fixed_touch_sequence  # noqa: N806
                    print(touchSequence)

                # Make sure the first touch will cause a movement.
                if (prevTouchSequence is not None and touchSequence[0]
                        == prevTouchSequence[-1]):
                    continue

                break

            sense_sequence = []  # contains a list of all the previous input SDRs
            prediction_sequence = []  # contains a list of the current SDR prediction,
            # as well as previously sensed input SDRs up until inference is successful
            progression = {}
            progression['classified'] = []
            progression['proportion'] = []
            progression['step'] = []

            for i_feature in touchSequence:
                currentStep += 1
                feature = objectDescription["features"][i_feature]

                self._move(feature, randomLocation=randomLocation)

                pre_touch_location_list = self.column.get_location_copy()  # Save
                # representation for later

                featureSDR = self.features[feature["name"]]  # noqa: N806

                self._sense(featureSDR, learn=False, waitForSettle=False)

                predictedColumns = map(int, list(set(np.floor(  # noqa: N806
                    self.column.L4.getBasalPredictedCells() / cellsPerColumn))))
                # Note _sense of the feature itself does not change the predicted
                # columns on this touch iteration (and thus does not invalidate the
                # prediction), but it does ensure the BasalPredictedCells have been
                # updated following the movement, and we re-set the location
                # representation later once in post-inference

                # Include all previously sensed/predicted representaitons by
                # over-writing current_sequence
                current_sequence = sense_sequence[:]

                current_sequence.append(list(predictedColumns))  # include the newly
                # predicted columns

                if currentStep == 1:  # On the first step, record the input sensation
                    prediction_sequence.append([featureSDR[:]])

                else:
                    prediction_sequence.append(current_sequence)

                if not inferred:
                    sense_sequence.append(featureSDR[:])

                else:
                    # Re-set location representations after inference successful so
                    # that additional sensations don't influence predictions, and we
                    # can use the output predictions to visualize what the network sees
                    # across the rest of the input space
                    module_iter = 0
                    for module in self.column.L6aModules:
                        module.activeCells = pre_touch_location_list[module_iter]
                        module_iter += 1

                    # Once inference has taken place, sense_sequence gathers
                    # predictions
                    sense_sequence.append(list(predictedColumns))

                if not inferred:
                    # Use the sensory-activated cells to detect whether the object has
                    # been recognized. If these sensory-activated cells
                    # are correct, it implies that the input layer's representation is
                    # classifiable -- the location layer just correctly classified it.

                    representation = \
                        self.column.getSensoryAssociatedLocationRepresentation()

                    classification_list = []

                    proportion = 0.0

                    # NB a minimum number of steps are required before inference takes place
                    # This reduces false positives in very early inference
                    if (len(set(representation)) > 0) and (currentStep >= 5):

                        # vector to store 1 where a location has been active
                        print("\n\n")
                        active_loc_vector = np.zeros(np.shape(self.class_weights)[0])
                        print(np.shape(active_loc_vector))
                        print("\n Current representations:")
                        print(np.shape(representation))
                        print(np.shape(list(set(representation))))
                        print("Max value:")
                        print(np.max(representation))
                        
                        active_loc_vector[representation] = 1

                        print("\nOutput:")
                        class_node_activations = np.matmul(active_loc_vector, self.class_weights)
                        print(np.shape(class_node_activations))
                        print(class_node_activations)
                        print("\nCurrent node-based classifiction is:" + str(np.argmax(class_node_activations)))
                        print("Ground truth: " + objectDescription["name"][0])


                        # Track the proportion by which the winning node is active, as well as whether it's the correct node
                        # This will give a flavour of an appropriate hyper-parameter value --> in particular it should capture true
                        # wins, while not being sensitive to false ones


                        proportion = np.max(class_node_activations)/np.sum(class_node_activations)

                        progression['classified'].append(np.argmax(class_node_activations) == int(objectDescription["name"][0]))
                        progression['proportion'].append(proportion)
                        progression['step'].append(currentStep)

                        # if (proportion >= 0.4) and (np.argmax(class_node_activations) == int(objectDescription["name"][0])):
                        #     print("Correctly classified")
                        #     inferredStep = currentStep  # noqa: N806
                        #     plt.imsave("correctly_classified/trial_" + str(trial_iter)
                        #                + "_" + objectDescription["name"]
                        #                + ".png", objectImage)
                        #     incorrect = {"never_converged": 0, "false_convergence": 0}

                        #     break

                        # # Check the representation against all possible target classes
                        # for target_iter in range(10):
                        #     target_representations = \
                        #         all_class_targets[target_iter][i_feature]

                        #     if (set(representation) <= target_representations):
                        #         classification_list.append(target_iter)
                        #         print("Classified as a " + str(target_iter)
                        #               + " with ground truth of "
                        #               + objectDescription["name"])

                    # if len(classification_list) > 1:
                    #     print("Classification list : classified as multiple classes!")
                    #     print(classification_list)

                    #     print("***Provisional code to handle multiple classes"
                    #           "implemented, but as never experienced, the reliability"
                    #           "of the intersection resolution is untested***")
                    #     exit()

                    #     intersection_list = []

                    #     # Check the amount of overlap between the representations of
                    #     for ambiguous_class_iter in range(len(classification_list)):

                    #         intersection_list.append(
                    #             len(set(representation).intersection(
                    #                 all_class_targets[
                    #                     classification_list[
                    #                         ambiguous_class_iter]][i_feature])))

                    #     # *** note this does not deal with Numpy's behaviour if two
                    #     # locations have the same maximum value -->
                    #     # see https://numpy.org/doc/stable/reference/
                    #     # generated/numpy.argmax.html

                    #     print("Intersection list:")
                    #     print(intersection_list)
                    #     classification_list = classification_list[
                    #         np.armax(intersection_list)]

                    #     print("After checking intersections, resolved to:")
                    #     print(classification_list)

                    inferred = (proportion >= class_threshold)

                    if inferred:
                        if np.argmax(class_node_activations) == int(objectDescription["name"][0]):
                            print("Correctly classified")
                            inferredStep = currentStep  # noqa: N806
                            # plt.imsave("correctly_classified/trial_" + str(trial_iter)
                            #            + "_" + objectDescription["name"]
                            #            + ".png", objectImage)
                            incorrect = {"never_converged": 0, "false_convergence": 0}

                        else:
                            print("Incorrectly classified")
                            incorrect = {"never_converged": 0, "false_convergence": 1}
                            # plt.imsave("misclassified/trial_" + str(trial_iter)
                            #            + "_example_" + objectDescription["name"]
                            #            + "_converged_to_" + str(classification_list[0])
                            #            + ".png", objectImage)
                            return None, incorrect, prediction_sequence, touchSequence, progression

                finished = ((((inferred and numSensations is None)
                            or (numSensations is not None and currentStep
                                == numSensations))) and currentStep == 25)
                # Continuing to step 25 ensures we gather network predictions even after
                # inference is successful

                if finished:
                    break

            prevTouchSequence = touchSequence  # noqa: N806

            if finished:
                break

        if incorrect["never_converged"] == 1:
            print("\nNever converged!")
            print("Inferred step when never converged " + str(inferredStep))

        return inferredStep, incorrect, prediction_sequence, touchSequence, progression
