# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2022, Numenta, Inc.  Unless you have an agreement
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

import random

import matplotlib
import matplotlib.pyplot as plt
import torch

from nupic.research.frameworks.htm import SequenceMemoryApicalTiebreak

matplotlib.use("Agg")

real_type = torch.float32
int_type = torch.int64

def accuracy(current, predicted):
    """
    Computes the accuracy of the TM at time-step t based on the prediction
    at time-step t-1 and the current active columns at time-step t.

    @param current (array) binary vector containing current active columns
    @param predicted (array) binary vector containing predicted active columns

    @return acc (float) prediction accuracy of the TM at time-step t
    """

    accuracy = 0
    if torch.count_nonzero(predicted) > 0:
        accuracy = float(torch.dot(current, predicted)) / float(
            torch.count_nonzero(predicted)
        )
    return accuracy


def corruptVector(v1, noiseLevel, numActiveCols):
    """
    Corrupts a copy of a binary vector by inverting noiseLevel percent of its bits.

    @param v1 (array) binary vector whose copy will be corrupted
    @param noiseLevel  (float) amount of noise to be applied on the new vector
    @param numActiveCols (int)   number of sparse columns that represent an input

    @return v2 (array) corrupted binary vector
    """

    size = len(v1)
    v2 = torch.zeros(size, dtype=int_type)
    bitsToSwap = int(noiseLevel * numActiveCols)

    # Copy the contents of v1 into v2
    for i in range(size):
        v2[i] = v1[i]
    for _ in range(bitsToSwap):
        i = random.randrange(size)
        if v2[i] == 1:
            v2[i] = 0
        else:
            v2[i] = 1
    return v2


def showPredictions():
    """
    Shows predictions of the TM when presented with the characters A, B, C, D, X, and
    Y without any contextual information, that is, not embedded within a sequence.
    """

    for k in range(6):
        tm.reset()
        print("--- " + "ABCDXY"[k] + " ---")

        tm.compute(seq_t[k][:].nonzero().squeeze(), learn=False)

        active_minicolumn_indices = [
            (i//tm.num_cells_per_minicolumn) for i in tm.get_active_cells().tolist()
        ]
        predicted_minicolumn_indices = [
            (i//tm.num_cells_per_minicolumn) for i in tm.get_predicted_cells().tolist()
        ]

        current_minicolumns = torch.Tensor([
            1 if i in active_minicolumn_indices else 0 \
                for i in range(tm.num_minicolumns)
        ])
        predicted_minicolumns = torch.Tensor([
            1 if i in predicted_minicolumn_indices else 0 \
                for i in range(tm.num_minicolumns)
        ])

        print("Active cols: " + str(torch.nonzero(current_minicolumns).squeeze()))
        print("Predicted cols: " + str(torch.nonzero(predicted_minicolumns).squeeze()))
        print()


def trainTM(sequence, timeSteps, noiseLevel):
    """
    Trains the TM with given sequence for a given number of time steps and level of
    input corruption

    @param sequence   (array) array whose rows are the input characters
    @param timeSteps  (int)   number of time steps in which the TM will be presented
    with sequence
    @param noiseLevel (float) amount of noise to be applied on the characters in the
    sequence
    """

    current_minicolumns = torch.zeros(tm.num_minicolumns, dtype=int_type)
    predicted_minicolumns = torch.zeros(tm.num_minicolumns, dtype=int_type)
    ts = 0

    for _ in range(timeSteps):
        tm.reset()

        for k in range(4):
            v = corruptVector(sequence[k][:], noiseLevel, sparse_cols)

            tm.compute(v[:].nonzero().squeeze(), learn=True)

            active_minicolumn_indices = [
                (i//tm.num_cells_per_minicolumn) \
                    for i in tm.get_active_cells().tolist()
            ]
            predicted_minicolumn_indices = [
                (i//tm.num_cells_per_minicolumn) \
                    for i in tm.get_predicted_cells().tolist()
            ]

            current_minicolumns = torch.Tensor([
                1 if i in active_minicolumn_indices else 0 \
                    for i in range(tm.num_minicolumns)
            ])
            predicted_minicolumns = torch.Tensor([
                1 if i in predicted_minicolumn_indices else 0 \
                    for i in range(tm.num_minicolumns)
            ])

            acc = accuracy(current_minicolumns, predicted_minicolumns)

            x.append(ts)
            y.append(acc)
            ts += 1


"""
A simple tutorial that shows some features of the Temporal Memory.
The following program has the purpose of presenting some
basic properties of the Temporal Memory, in particular when it comes
to how it handles high-order sequences.
"""

tm = SequenceMemoryApicalTiebreak(
    num_minicolumns=2048,
    num_cells_per_minicolumn=8,
    initial_permanence=0.21,
    connected_permanence=0.3,
    matching_threshold=15,
    permanence_increment=0.1,
    permanence_decrement=0.1,
    activation_threshold=15,
    basal_segment_incorrect_decrement=0.01
)

random.seed(1)

sparsity = 0.02
sparse_cols = int(tm.num_minicolumns * sparsity)

# We will create a sparse representation of characters A, B, C, D, X, and Y.
# In this particular example we manually construct them, but usually you would
# use the spatial pooler to build these.
seq1 = torch.zeros((4, tm.num_minicolumns), dtype=int_type)
seq1[0, 0:sparse_cols] = 1  # Input SDR representing "A"
seq1[1, sparse_cols : 2 * sparse_cols] = 1  # Input SDR representing "B"
seq1[2, 2 * sparse_cols : 3 * sparse_cols] = 1  # Input SDR representing "C"
seq1[3, 3 * sparse_cols : 4 * sparse_cols] = 1  # Input SDR representing "D"

seq2 = torch.zeros((4, tm.num_minicolumns), dtype=int_type)
seq2[0, 4 * sparse_cols : 5 * sparse_cols] = 1  # Input SDR representing "X"
seq2[1, sparse_cols : 2 * sparse_cols] = 1  # Input SDR representing "B"
seq2[2, 2 * sparse_cols : 3 * sparse_cols] = 1  # Input SDR representing "C"
seq2[3, 5 * sparse_cols : 6 * sparse_cols] = 1  # Input SDR representing "Y"

seq_t = torch.zeros((6, tm.num_minicolumns), dtype=int_type)
seq_t[0, 0:sparse_cols] = 1  # Input SDR representing "A"
seq_t[1, sparse_cols : 2 * sparse_cols] = 1  # Input SDR representing "B"
seq_t[2, 2 * sparse_cols : 3 * sparse_cols] = 1  # Input SDR representing "C"
seq_t[3, 3 * sparse_cols : 4 * sparse_cols] = 1  # Input SDR representing "D"
seq_t[4, 4 * sparse_cols : 5 * sparse_cols] = 1  # Input SDR representing "X"
seq_t[5, 5 * sparse_cols : 6 * sparse_cols] = 1  # Input SDR representing "Y"

# PART 1. Feed the TM with sequence "ABCD". The TM will eventually learn
# the pattern and it's prediction accuracy will go to 1.0 (except in-between sequences
# where the TM doesn't output any prediction)
print()
print("-" * 50)
print(
    "Part 1. We present the sequence ABCD to the TM. The TM will eventually\n \
    will learn the sequence and predict the upcoming characters. This can be\n \
    measured by the prediction accuracy in Fig 1. Note that in-between sequences\n \
    the accuracy is 0.0 as the TM does not output any prediction."
)
print("-" * 50)
print()

x = []
y = []

trainTM(seq1, timeSteps=10, noiseLevel=0.0)

plt.ylim([-0.1, 1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 1: TM learns sequence ABCD")
plt.savefig("figure_1")
plt.close()

print()
print("-" * 50)
print(
    "Once the TM has learned the sequence ABCD, we present the individual\n \
    characters to the TM to know its prediction. The TM outputs the columns\n \
    that become active upon the presentation of a particular character as well\n \
    as the columns predicted in the next time step. You should see that A\n \
    predicted B, B predicts C, C predicts D, and D does not output any\n \
    prediction. Note that we present individual characters -- a single character\n \
    deprived of context in a sequence. There is no prediction for characters\n \
    X and Y as we have not yet presented them to TM in any sequence."
)
print("-" * 50)
print()

showPredictions()

print()
print("-" * 50)
print(
    "Part 2: Present the sequence XBCY to TM. As expected, accuracy will drop until\n \
    TM learns the new sequence (Fig 2). What will be the prediction of the TM if\n \
    presented with the sequence BC? This would depend on what character precedes B."
)
print("-" * 50)
print()

x = []
y = []

trainTM(seq2, timeSteps=10, noiseLevel=0.0)

# In this figure you can see how the TM starts making good predictions for particular
# characters (spikes in the plot). Then, it will get half of its predictions right,
# which correspond to the times in which is presented with character C. After some
# time, it will learn correctly the sequence XBCY, and predict its characters
# accordingly.
plt.ylim([-0.1, 1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 2: TM learns new sequence XBCY")
plt.savefig("figure_2")
plt.close()

print()
print("-" * 50)
print(
    "We will present again each of the two characters individually to the TM.\n \
    That is, not within any of the two sequences. When presented with the\n \
    character A, the TM predicts B, B predicts C, and C outputs a simultaneous\n \
    prediction of both D and Y. In order to disambiguate, the TM would require\n \
    to know if the preceding characters were AB or XB. When presented with\n \
    the character X, the TM predicts B, whereas Y and D yield no prediction."
)
print("-" * 50)
print()

showPredictions()

# PART 3. Now we will present noisy inputs to the TM. We will add noise to the
# sequence XBCY by corrupting 30% of its bits. We would like to see how the TM responds
# in the presence of noise and how it recovers from it.
print()
print("-" * 50)
print(
    "Part 3: We will add noise to the sequence XBCY by corrupting 30% of the bits\n \
    in the vectors encoding each character. Expect to see a decrease in prediction\n \
    accuracy as the TM is unable to learn the random noise in the input (Fig. 3).\n \
    However, decrease is not significant."
)
print("-" * 50)
print()

x = []
y = []

trainTM(seq2, timeSteps=50, noiseLevel=0.3)

plt.ylim([-0.1, 1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 3: Accuracy in TM with 30% noise in input")
plt.savefig("figure_3")
plt.close()

print()
print("-" * 50)
print(
    "Take a look at the output of the TM when presented with noisy input\n \
    (30%). Noise is low so TM is not affected by it. It would be affected\n \
    if we saw noisy columns being predicted when presented with individual\n \
    characters. Thus, TM exhibits resilience to noise in its input."
)
print("-" * 50)
print()

showPredictions()

# Let's corrupt the sequence more by adding 50% of noise to each of its characters.
# Here, we would expect to see some 'noisy' columns being predicted when the TM is
# presented with the individual characters.

print()
print("-" * 50)
print(
    "Now, we will set noise to be 50% of the bits in the characters X, B, C, and Y.\n \
    The accuracy will decrease (Fig. 5) and noisy columns will be predicted by the TM."
)
print("-" * 50)
print()

x = []
y = []

trainTM(seq2, timeSteps=50, noiseLevel=0.5)

print()
print("-" * 50)
print(
    "The prediction of some characters (e.g. X) now includes columns that are\n \
    not related to any other character. This is because the TM tried to learn the\n \
    noise in the input patterns."
)
print("-" * 50)
print()

showPredictions()

plt.ylim([-0.1, 1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 4: Accuracy in TM with 50% noise in input")
plt.savefig("figure_4")
plt.close()

# Will the TM be able to forget the 'noisy' columns learned in the previous step?
# We will present the TM with the original sequence XBCY so it forgets the 'noisy'.
# columns.

x = []
y = []

trainTM(seq2, timeSteps=10, noiseLevel=0.0)

print()
print("-" * 50)
print(
    "After presenting the original sequence XBCY to the TM, we expect to see the\n \
    predicted noisy columns from the previous step disappear. We verify that by\n \
    presenting the individual characters to the TM."
)
print("-" * 50)
print()

showPredictions()

# We can see how the prediction accuracy goes back to 1.0 (as before,
# not in-between sequences) when the TM 'forgets' the noisy columns.
plt.ylim([-0.1, 1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 5: TM forgets noise in sequence XBCY when noise is over")
plt.savefig("figure_5")
plt.close()

# Let's corrupt the sequence even more and add 90% of noise to each of its characters.
# Here, we would expect to see even more of a decrease in accuracy along with more
# 'noisy' columns being predicted.
print()
print("-" * 50)
print(
    "We will add more noise to the characters in the sequence XBCY. We now corrupt\n \
    90% of its contents. The accuracy will decrease (Fig. 6) and noisy columns\n \
    will be predicted by the TM."
)
print("-" * 50)
print()

x = []
y = []

trainTM(seq2, timeSteps=50, noiseLevel=0.9)

print()
print("-" * 50)
print(
    "Look at output when presented with individual characters of sequence.\n \
    Noisy predicted columns emerge as a result of the TM trying to learn the\n \
    noise."
)
print("-" * 50)
print()

showPredictions()

# In this figure we can observe how the prediction accuracy is affected by the presence
# of noise in the input. However, the accuracy does not drops dramatically even with 90%
# of noise which implies that the TM exhibits some resilience to noise in its input
# which means that it does not forget easily a well-learned, real pattern.
plt.ylim([-0.1, 1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 6: Accuracy with 90% noise in input")
plt.savefig("figure_6")
plt.close()

# Let's present the original sequence to the TM in order to make it forget the noisy
# columns. After this, the TM will predict accurately the sequence again, and its
# predictions will not include 'noisy' columns anymore.
x = []
y = []

trainTM(seq2, timeSteps=25, noiseLevel=0.0)

# We will observe how the prediction accuracy gets back to 1.0 (not in-between
# sequences) as the TM is presented with the original sequence.
plt.ylim([-0.1, 1.1])
plt.plot(x, y)
plt.xlabel("Timestep")
plt.ylabel("Prediction Accuracy")
plt.title("Fig. 7: When noise is suspended, accuracy is restored")
plt.savefig("figure_7")
plt.close()

# The TM restores its prediction accuracy and it can be seen when presented with the
# individual characters. There's no noisy columns being predicted.
print()
print("-" * 50)
print(
    "After presenting noisy input to the TM, we present the original sequence\n \
    in order to make it re-learn XBCY. Verify by presenting the TM with the\n \
    individual characters and observing its output. Again, we can see that noisy\n \
    columns are not being predicted anymore, and that the prediction accuracy goes\n \
    back to 1.0 when the sequence is presented (Fig. 7)."
)
print("-" * 50)
print()

showPredictions()

# PART 4. Now, we will present both sequences ABCD and XBCY randomly to the TM.
# For this purpose we will start with a new TM.
# What would be the output of the TM when presented with character D if it has
# been exposed to sequences ABCD and XBCY occurring randomly one after the other?
# If one quarter of the time the TM sees the sequence ABCDABCD, another quarter the
# TM sees ABCDXBCY, another quarter it sees XBCYXBCY, and the last quarter it saw
# XBCYABCD, then the TM would exhibit simultaneous predictions for characters D, Y
# and C.
print()
print("-" * 50)
print(
    "Part 4: Present both ABCD and XBCY randomly to the TM. We might observe\n \
    simultaneous predictions occurring when TM is presented with D, Y, and C.\n \
    Use a blank TM for this reason. Note that we don't reset the TM after\n \
    presenting each sequence since we want TM to learn different predictions\n \
    for D and Y."
)
print("-" * 50)
print()

tm = SequenceMemoryApicalTiebreak(
    num_minicolumns=2048,
    num_cells_per_minicolumn=8,
    initial_permanence=0.21,
    connected_permanence=0.3,
    matching_threshold=15,
    permanence_increment=0.1,
    permanence_decrement=0.1,
    activation_threshold=15,
    basal_segment_incorrect_decrement=0.01
)

for _ in range(75):
    rnd = random.randrange(2)
    for k in range(4):
        if rnd == 0:
            tm.compute(seq1[k][:].nonzero().squeeze(), learn=True)
        else:
            tm.compute(seq2[k][:].nonzero().squeeze(), learn=True)

print()
print("-" * 50)
print(
    "Look at output when presented with A, B, C, D, X, and Y. We might observe\n \
    simultaneous predictions when presented with D (predicting A and X), with\n \
    character Y (predicting A and X), and with C (predicting D and Y). Note, due to\n \
    stochasticity of this script, we might not observe simultaneous predictions\n \
    in *all* aforementioned characters."
)
print("-" * 50)
print()

showPredictions()
