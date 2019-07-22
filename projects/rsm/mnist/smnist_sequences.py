#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import numpy as np
import torch


def generate_subsequences(start_digits=None, digits=10, length=4):
    if start_digits is None:
        start_digits = [0, 1]
    seq = torch.zeros(digits, length)
    reps = int(np.ceil(digits / len(start_digits)))
    first_col = torch.repeat_interleave(torch.tensor(start_digits), reps, 0)
    seq[:, 0] = first_col[:digits]
    for i in range(1, length):
        column = torch.arange(digits)
        idxs = torch.randperm(digits)
        seq[:, i] = column[idxs]
    print(seq)


generate_subsequences(digits=6)
