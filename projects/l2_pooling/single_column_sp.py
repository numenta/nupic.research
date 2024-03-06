# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
This file creates simple experiments to test a single column L4-L2 network.
"""

from nupic.research.frameworks.columns.object_machine_factory import (
  createObjectMachine
)
from nupic.research.frameworks.columns.l2_l4_inference import L4L2Experiment



def createThreeObjects():
  """
  Helper function that creates a set of three objects used for basic
  experiments.

  :return:   (list(list(tuple))  List of lists of feature / location pairs.
  """
  objectA = list(zip(list(range(10)), list(range(10))))
  objectB = [(0, 0), (2, 2), (1, 1), (1, 4), (4, 2), (4, 1)]
  objectC = [(0, 0), (1, 1), (3, 1), (0, 1)]
  return [objectA, objectB, objectC]



def runSharedFeatures(noiseLevel=None):
  """
  Runs a simple experiment where three objects share a number of location,
  feature pairs.

  Parameters:
  ----------------------------
  @param    noiseLevel (float)
            Noise level to add to the locations and features during inference

  """
  exp = L4L2Experiment(
    "shared_features",
    enableLateralSP=True,
    enableFeedForwardSP=True
  )

  pairs = createThreeObjects()
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024
  )
  for object in pairs:
    objects.addObject(object)

  exp.learnObjects(objects.provideObjectsToLearn())

  inferConfig = {
    "numSteps": 10,
    "noiseLevel": noiseLevel,
    "pairs": {
      0: list(zip(list(range(10)), list(range(10))))
    }
  }

  exp.infer(objects.provideObjectToInfer(inferConfig), objectName=0)

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"],
  )



def runUncertainLocations(missingLoc=None):
  """
  Runs the same experiment as above, with missing locations at some timesteps
  during inference (if it was not successfully computed by the rest of the
  network for example).

  @param   missingLoc (dict)
           A dictionary mapping indices in the object to location index to
           replace with during inference (-1 means no location, a tuple means
           an union of locations).

  """
  if missingLoc is None:
    missingLoc = {}

  exp = L4L2Experiment(
    "uncertain_location",
    enableLateralSP = True,
    enableFeedForwardSP=True
  )

  pairs = createThreeObjects()
  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024
  )
  for object in pairs:
    objects.addObject(object)

  exp.learnObjects(objects.provideObjectsToLearn())

  # create pairs with missing locations
  objectA = objects[0]
  for key, val in missingLoc.items():
    objectA[key] = (val, key)

  inferConfig = {
    "numSteps": 10,
    "pairs": {
      0: objectA
    }
  }

  exp.infer(objects.provideObjectToInfer(inferConfig), objectName=0)

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation",
            "L4 Predictive"],
  )



def runStretchExperiment(numObjects=25):
  """
  Generates a lot of random objects to profile the network.

  Parameters:
  ----------------------------
  @param    numObjects (int)
            Number of objects to create and learn.

  """
  exp = L4L2Experiment(
    "profiling_experiment",
    enableLateralSP = True,
    enableFeedForwardSP=True
  )

  objects = createObjectMachine(
    machineType="simple",
    numInputBits=20,
    sensorInputSize=1024,
    externalInputSize=1024
  )
  objects.createRandomObjects(numObjects=numObjects, numPoints=10)
  exp.learnObjects(objects.provideObjectsToLearn())

  inferConfig = {
    "numSteps": len(objects[0]),
    "pairs": {
      0: objects[0]
    }
  }

  exp.infer(objects.provideObjectToInfer(inferConfig), objectName=0)

  exp.plotInferenceStats(
    fields=["L2 Representation",
            "Overlap L2 with object",
            "L4 Representation"]
  )



if __name__ == "__main__":
  # basic experiment with shared features
  runSharedFeatures()

  # experiment with unions at locations
  missingLoc = {3: (1,2,3), 6: (6,4,2)}
  runUncertainLocations(missingLoc=missingLoc)

  # stretch experiment to profile the regions
  runStretchExperiment()
