# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
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

"""TODO"""

import collections
import json

import numpy as np


def generateObjects(numObjects, numFeatures):
    objects = {}
    for i in range(numObjects):
        obj = np.random.randint(numFeatures, size=10, dtype=np.int32)
        objects[i] = obj
    return objects


def runTrial(objects):
    results = collections.defaultdict(int)

    objectSets = []
    for targetID, targetObj in objects.items():
        objectSets.append((targetID, frozenset(targetObj)))

    for _targetID, targetObj in objects.items():
        np.random.shuffle(targetObj)

        candidates = objectSets
        for i in range(len(targetObj)):
            step = i + 1
            feats = frozenset(targetObj[:step])
            newCandidates = []
            for objID, obj in candidates:
                if feats <= obj:
                    newCandidates.append((objID, obj))
            candidates = newCandidates
            if len(candidates) == 1:
                results[step] += 1
                break
        else:
            results[None] += 1

    return results


def runSim(numObjects, numFeatures, numTrials):
    # List of trials, each a map from recognition time to number of occurrences
    results = []

    for _ in range(numTrials):
        objects = generateObjects(numObjects, numFeatures)
        results.append(runTrial(objects))

    with open("results/bof.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    runSim(100, 10, 10)
