import numpy as np
import torch


def generate_subsequences(start_digits=[0, 1], digits=10, length=4):
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