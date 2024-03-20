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
This evaluates the effect of synapse sampling on the feedforward and lateral
connections of L2. Specifically, how low can we go with L2 activation threshold,
number of distal synapses and number of proximal synapses while still get
reliable performance.

We consider the problem of multi-column convergence.
"""

import pickle
import random
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nupic.research.frameworks.columns.l2_l4_inference import L4L2Experiment
from nupic.research.frameworks.columns.object_machine_factory import createObjectMachine

plt.ion()


def getL4Params():
    """
    Returns a good default set of parameters to use in the L4 region.
    """
    return {
        "columnCount": 2048,
        "cellsPerColumn": 8,
        "initialPermanence": 0.51,
        "connectedPermanence": 0.6,
        "permanenceIncrement": 0.1,
        "permanenceDecrement": 0.02,
        "minThreshold": 10,
        "basalPredictedSegmentDecrement": 0.002,
        "activationThreshold": 13,
        "sampleSize": 20,
        "seed": 41,
    }


def getL2Params():
    """
    Returns a good default set of parameters to use in the L4 region.
    """
    return {
        "inputWidth": 2048 * 8,
        "cellCount": 4096,
        "sdrSize": 40,
        "synPermProximalInc": 0.1,
        "synPermProximalDec": 0.001,
        "initialProximalPermanence": 0.6,
        "minThresholdProximal": 1,
        "sampleSizeProximal": 20,
        "connectedPermanenceProximal": 0.5,
        "synPermDistalInc": 0.1,
        "synPermDistalDec": 0.001,
        "initialDistalPermanence": 0.41,
        "activationThresholdDistal": 13,
        "sampleSizeDistal": 20,
        "connectedPermanenceDistal": 0.5,
        "seed": 41,
    }


def runExperiment(args):
    """
    Run experiment.  What did you think this does?

    args is a dict representing the parameters. We do it this way to support
    multiprocessing. args contains one or more of the following keys:

    @param noiseLevel  (float) Noise level to add to the locations and features
                               during inference. Default: None
    @param numObjects  (int)   The number of objects we will train.
                               Default: 10
    @param numPoints   (int)   The number of points on each object.
                               Default: 10
    @param numLocations (int)  For each point, the number of locations to choose
                               from.  Default: 10
    @param numFeatures (int)   For each point, the number of features to choose
                               from.  Default: 10
    @param numColumns  (int)   The total number of cortical columns in network.
                               Default: 2

    The method returns the args dict updated with two additional keys:
      convergencePoint (int)   The average number of iterations it took
                               to converge across all objects
      objects          (pairs) The list of objects we trained on
    """
    numObjects = args.get("numObjects", 10)
    numLocations = args.get("numLocations", 10)
    numFeatures = args.get("numFeatures", 10)
    numColumns = args.get("numColumns", 2)
    # noiseLevel = args.get("noiseLevel", None)  # TODO: implement this?
    numPoints = args.get("numPoints", 10)
    trialNum = args.get("trialNum", 42)
    l2Params = args.get("l2Params", getL2Params())
    l4Params = args.get("l4Params", getL4Params())
    objectSeed = args.get("objectSeed", 41)

    # Create the objects
    objects = createObjectMachine(
        machineType="simple",
        numInputBits=20,
        sensorInputSize=1024,
        externalInputSize=1024,
        numCorticalColumns=numColumns,
        seed=objectSeed,
    )
    objects.createRandomObjects(
        numObjects,
        numPoints=numPoints,
        numLocations=numLocations,
        numFeatures=numFeatures,
    )

    # print "Objects are:"
    # for o in objects:
    #   pairs = objects[o]
    #   pairs.sort()
    #   print str(o) + ": " + str(pairs)

    # Setup experiment and train the network
    name = "convergence_O%03d_L%03d_F%03d_C%03d_T%03d" % (
        numObjects,
        numLocations,
        numFeatures,
        numColumns,
        trialNum,
    )
    exp = L4L2Experiment(
        name,
        L2Overrides=l2Params,
        L4Overrides=l4Params,
        numCorticalColumns=numColumns,
        seed=trialNum,
    )

    exp.learnObjects(objects.provideObjectsToLearn())

    # For inference, we will check and plot convergence for each object. For each
    # object, we create a sequence of random sensations for each column.  We will
    # present each sensation for 3 time steps to let it settle and ensure it
    # converges.

    for objectId in objects:
        obj = objects[objectId]

        # Create sequence of sensations for this object for all columns
        objectSensations = {}
        for c in range(numColumns):
            objectCopy = [pair for pair in obj]
            random.shuffle(objectCopy)
            # stay multiple steps on each sensation
            sensations = []
            for pair in objectCopy:
                for _ in range(2):
                    sensations.append(pair)
            objectSensations[c] = sensations

        inferConfig = {
            "object": objectId,
            "numSteps": len(objectSensations[0]),
            "pairs": objectSensations,
        }

        exp.infer(objects.provideObjectToInfer(inferConfig), objectName=objectId)

    convergencePoint, _ = exp.averageConvergencePoint("L2 Representation", 40, 40)
    print(
        "objectSeed {} # distal syn {} # proximal syn {}, "
        "# convergence point={:4.2f}".format(
            objectSeed,
            l2Params["sampleSizeDistal"],
            l2Params["sampleSizeProximal"],
            convergencePoint,
        )
    )

    # Return our convergence point as well as all the parameters and objects
    args.update({"objects": objects.getObjects()})
    args.update({"convergencePoint": convergencePoint})

    # prepare experiment results
    numLateralConnections = []
    numProximalConnections = []
    for l2Columns in exp.L2Columns:
        numLateralConnections.append(l2Columns.numberOfDistalSynapses())
        numProximalConnections.append(np.sum(l2Columns.numberOfProximalSynapses()))

    result = {
        "trial": objectSeed,
        "sampleSizeProximal": l2Params["sampleSizeProximal"],
        "sampleSizeDistal": l2Params["sampleSizeDistal"],
        "numLateralConnections": np.mean(np.array(numLateralConnections)),
        "numProximalConnections": np.mean(np.array(numProximalConnections)),
        "convergencePoint": args["convergencePoint"],
    }
    return result


def experimentVaryingSynapseSampling(
    expParams, sampleSizeDistalList, sampleSizeProximalList
):
    """
    Test multi-column convergence with varying amount of proximal/distal sampling

    :return:
    """
    numRpts = 20
    args = []
    for sampleSizeProximal in sampleSizeProximalList:
        for sampleSizeDistal in sampleSizeDistalList:

            for rpt in range(numRpts):
                l4Params = getL4Params()
                l2Params = getL2Params()
                l2Params["sampleSizeProximal"] = sampleSizeProximal
                l2Params["minThresholdProximal"] = sampleSizeProximal
                l2Params["sampleSizeDistal"] = sampleSizeDistal
                l2Params["activationThresholdDistal"] = sampleSizeDistal

                args.append(
                    {
                        "numObjects": expParams["numObjects"],
                        "numLocations": expParams["numLocations"],
                        "numFeatures": expParams["numFeatures"],
                        "numColumns": expParams["numColumns"],
                        "trialNum": rpt,
                        "l4Params": l4Params,
                        "l2Params": l2Params,
                        "objectSeed": rpt,
                    }
                )

    use_pool = False
    if use_pool:
        pool = Pool(processes=expParams["numWorkers"])
        result = pool.map(runExperiment, args)
    else:
        result = []
        for arg in args:
            result.append(runExperiment(arg))

    return result


def convertResultsToDataFrames(results):
    df = None
    for result in results:
        if df is None:
            df = pd.DataFrame.from_dict(result, orient="index")
        else:
            df = pd.concat([df, pd.DataFrame.from_dict(result, orient="index")], axis=1)
    df = df.transpose()
    return df


def experimentVaryingDistalSynapseNumber(expParams):
    sampleSizeDistalList = [2, 3, 4, 5, 6, 8, 10, 15, 20]
    sampleSizeProximalList = [5]
    result = experimentVaryingSynapseSampling(
        expParams, sampleSizeDistalList, sampleSizeProximalList
    )

    resultsName = (
        "./results/multi_column_distal_sampling_"
        "numFeature_{}_numColumn_{}".format(
            expParams["numFeatures"], expParams["numColumns"]
        )
    )
    with open(resultsName, "wb") as f:
        pickle.dump(result, f)
    return result


def experimentVaryingProximalSynapseNumber(expParams):
    """
    Fix distal synapse sampling, varying proximal synapse sampling
    :param expParams:
    :return:
    """
    sampleSizeDistalList = [5]
    sampleSizeProximalList = [1, 2, 3, 4, 5, 6, 8, 10, 15]
    result = experimentVaryingSynapseSampling(
        expParams, sampleSizeDistalList, sampleSizeProximalList
    )
    resultsName = (
        "./results/multi_column_proximal_sampling_"
        "numFeature_{}_numColumn_{}".format(
            expParams["numFeatures"], expParams["numColumns"]
        )
    )
    with open(resultsName, "wb") as f:
        pickle.dump(result, f)


def plotDistalSynSamplingResult(expParams):
    fig, ax = plt.subplots(2, 2)
    legends = []
    for numColumns in [3, 5, 7]:
        resultsName = (
            "./results/multi_column_distal_sampling_"
            "numFeature_{}_numColumn_{}".format(expParams["numFeatures"], numColumns)
        )
        with open(resultsName, "rb") as f:
            results = pickle.load(f)

        df = convertResultsToDataFrames(results)

        convergencePointList = []
        numLateralConnectionsList = []

        sampleSizeDistalList = np.sort(np.unique(df["sampleSizeDistal"]))
        sampleSizeProximalList = np.sort(np.unique(df["sampleSizeProximal"]))
        for sampleSizeDistal in sampleSizeDistalList:
            idx = np.where(
                np.logical_and(
                    df["sampleSizeDistal"] == sampleSizeDistal,
                    df["sampleSizeProximal"] == sampleSizeProximalList[0],
                )
            )[0]

            convergencePointList.append(np.mean(df["convergencePoint"].iloc[idx]))
            numLateralConnectionsList.append(
                np.mean(df["numLateralConnections"].iloc[idx])
            )

        ax[0, 0].plot(
            sampleSizeDistalList,
            convergencePointList,
            "-o",
            label="numColumn_{}".format(numColumns),
        )
        ax[0, 0].set_ylabel("# pts to converge")
        ax[0, 0].set_xlabel("Distal sample size")

        ax[0, 1].plot(sampleSizeDistalList, numLateralConnectionsList, "-o")
        ax[0, 1].set_ylabel("# lateral connections / column")
        ax[0, 1].set_xlabel("Distal sample size")

        # ax[1, 0].plot(sampleSizeDistalList, l2LearnTimeList, '-o')
        # ax[1, 0].set_ylabel('L2 training time (s)')
        # ax[1, 0].set_xlabel('Distal sample size')

        # ax[1, 1].plot(sampleSizeDistalList, l2InferTimeList, '-o')
        # ax[1, 1].set_ylabel('L2 infer time (s)')
        # ax[1, 1].set_xlabel('Distal sample size')

        legends.append("{}-column".format(numColumns))
    plt.tight_layout()
    ax[0, 0].set_title("distal synapse sampling")
    plt.legend(legends)
    plt.savefig("plots/L2PoolingDistalSynapseSampling.pdf")


def plotProximalSynSamplingResult(expParams):
    fig, ax = plt.subplots(2, 2)
    legends = []
    for numColumns in [3, 5, 7]:
        resultsName = (
            "./results/multi_column_proximal_sampling_"
            "numFeature_{}_numColumn_{}".format(expParams["numFeatures"], numColumns)
        )
        with open(resultsName, "rb") as f:
            results = pickle.load(f)

        df = convertResultsToDataFrames(results)

        convergencePointList = []
        numProximalConnectionsList = []

        sampleSizeDistalList = np.sort(np.unique(df["sampleSizeDistal"]))
        sampleSizeProximalList = np.sort(np.unique(df["sampleSizeProximal"]))
        for sampleSizeProximal in sampleSizeProximalList:
            idx = np.where(
                np.logical_and(
                    df["sampleSizeDistal"] == sampleSizeDistalList[0],
                    df["sampleSizeProximal"] == sampleSizeProximal,
                )
            )[0]

            convergencePointList.append(np.mean(df["convergencePoint"].iloc[idx]))
            numProximalConnectionsList.append(
                np.mean(df["numProximalConnections"].iloc[idx])
            )

        ax[0, 0].plot(
            sampleSizeProximalList,
            convergencePointList,
            "-o",
            label="numColumn_{}".format(numColumns),
        )
        ax[0, 0].set_ylabel("# pts to converge")
        ax[0, 0].set_xlabel("Proximal sample size")

        ax[0, 1].plot(sampleSizeProximalList, numProximalConnectionsList, "-o")
        ax[0, 1].set_ylabel("# proximal connections / column")
        ax[0, 1].set_xlabel("Proximal sample size")

        # ax[1, 0].plot(sampleSizeProximalList, l2LearnTimeList, '-o')
        # ax[1, 0].set_ylabel('L2 training time (s)')
        # ax[1, 0].set_xlabel('Proximal sample size')

        # ax[1, 1].plot(sampleSizeProximalList, l2InferTimeList, '-o')
        # ax[1, 1].set_ylabel('L2 infer time (s)')
        # ax[1, 1].set_xlabel('Proximal sample size')

        legends.append("{}-column".format(numColumns))
    plt.tight_layout()
    ax[0, 0].set_title("proximal synapse sampling")
    plt.legend(legends)
    plt.savefig("plots/L2PoolingProximalSynapseSampling.pdf")


def main():
    expParams = {
        "numObjects": 10,
        "numLocations": 10,
        "numFeatures": 3,
        "numColumns": 3,
        "numWorkers": 6,
    }

    for numColumns in [3, 5, 7]:
        expParams["numColumns"] = numColumns

        # Fixed number of proximal synapses, varying distal synapse sampling
        experimentVaryingDistalSynapseNumber(expParams)

        # Fixed number of distal synapses, varying distal synapse sampling
        experimentVaryingProximalSynapseNumber(expParams)

    plotDistalSynSamplingResult(expParams)
    plotProximalSynSamplingResult(expParams)


if __name__ == "__main__":
    main()
