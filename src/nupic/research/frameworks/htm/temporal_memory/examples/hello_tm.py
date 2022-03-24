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

from itertools import count

import torch

from nupic.research.frameworks.htm import SequenceMemoryApicalTiebreak

real_type = torch.float32
int_type = torch.int64

print("""
This program shows how to access the Temporal Memory directly by demonstrating
how to create a TM instance, train it with vectors, get predictions, and
inspect the state. The code here runs a very simple version of sequence learning,
with one cell per column. The TM is trained with the simple sequence A->B->C->D->E.
HOMEWORK: once you have understood exactly what is going on here, try changing
cellsPerColumn to 4. What is the difference between once cell per column and 4
cells per column?
""")


def formatRow(x):
    s = ""
    for c in range(len(x)):
        if c > 0 and c % 10 == 0:
            s += " "
        s += str(x[c])
    s += " "
    return s


# create TM with appropriate parameters
tm = SequenceMemoryApicalTiebreak(
    num_minicolumns=50,
    num_cells_per_minicolumn=2,
    initial_permanence=0.5,
    matching_threshold=8,
    permanence_increment=0.1,
    permanence_decrement=0.0,
    activation_threshold=8,
)

# create input vectors that are fed to TM
x = torch.zeros((5, tm.num_minicolumns), dtype=int_type)
x[0, 0:10] = 1  # Input SDR representing "A", corresponding to columns 0-9
x[1, 10:20] = 1  # Input SDR representing "B", corresponding to columns 10-19
x[2, 20:30] = 1  # Input SDR representing "C", corresponding to columns 20-29
x[3, 30:40] = 1  # Input SDR representing "D", corresponding to columns 30-39
x[4, 40:50] = 1  # Input SDR representing "E", corresponding to columns 40-49

# TRAIN TM: send sequence to TM for learning (repeat the sequence 10 times)
for i in range(5):

    # send each letter in sequence in order
    for j in range(5):
        active_minicolumns = torch.Tensor(
            list(set([i for i, j in zip(count(), x[j]) if j == 1]))
        )

        # compute method performs one step of learning and/or inference.
        tm.compute(active_minicolumns, learn=True)

        # The following print statements prints out the active cells, predictive
        # cells, active segments and winner cells.
        print("seeing pattern", ["A", "B", "C", "D", "E"][j])
        print("active cells ", tm.get_active_cells().tolist())
        print("predictive cells ", tm.get_predicted_cells().tolist())
        print("winner cells ", tm.get_learning_cells().tolist())
        print("# of active segments ", tm.get_num_basal_segments())
        print()

    tm.reset()
    print("Reset!")

# INFERENCE TM: send same sequences and observe outputs
for j in range(5):
    print("\n\n--------", "ABCDE"[j], "-----------")
    print("Raw input vector : " + formatRow(x[j].tolist()))

    active_minicolumns = torch.Tensor(
        list(set([i for i, j in zip(count(), x[j]) if j == 1]))
    )

    # Send each vector to the TM, with learning turned off
    tm.compute(active_minicolumns, learn=False)

    # The following print statements prints out the active cells, predictive
    # cells, active segments and winner cells.
    #
    # What you should notice is that the columns where active state is 1
    # represent the SDR for the current input pattern and the columns where
    # predicted state is 1 represent the SDR for the next expected pattern
    print("\nAll the active and predicted cells:")
    print("active cells ", tm.get_active_cells().tolist())
    print("predictive cells ", tm.get_predicted_cells().tolist())
    print("winner cells ", tm.get_learning_cells().tolist())
    print("# of active segments ", tm.get_num_basal_segments())

    active_minicolumn_indices = [
        int(i / tm.num_cells_per_minicolumn) for i in tm.get_active_cells().tolist()
    ]
    predicted_minicolumn_indices = [
        int(i / tm.num_cells_per_minicolumn) for i in tm.get_predicted_cells().tolist()
    ]

    # Reconstructing the active and inactive columns with 1 as active and 0 as
    # inactive representation.
    active_minicolumn_state = [
        "1" if i in active_minicolumn_indices else "0"
        for i in range(tm.num_minicolumns)
    ]
    active_minicolumn_str = "".join(active_minicolumn_state)

    predictive_minicolumn_state = [
        "1" if i in predicted_minicolumn_indices else "0"
        for i in range(tm.num_minicolumns)
    ]
    predictive_minicolumn_str = "".join(predictive_minicolumn_state)

    print("Active columns:    " + formatRow(active_minicolumn_str))
    print("Predicted columns: " + formatRow(predictive_minicolumn_str))
