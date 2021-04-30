import torch
import torch.nn as nn
import os

def genOrthgonal(dim):
    a = torch.zeros((dim, dim)).normal_(0, 1)
    q, r = torch.qr(a)
    d = torch.diag(r, 0).sign()
    diag_size = d.size(0)
    d_exp = d.view(1, diag_size).expand(diag_size, diag_size)
    q.mul_(d_exp)
    return q


def makeDeltaOrthogonal(weights, gain):
    rows = weights.size(0)
    cols = weights.size(1)
    if rows > cols:
        print("In_filters should not be greater than out_filters.")
    weights.data.fill_(0)
    dim = max(rows, cols)
    q = genOrthgonal(dim)
    mid1 = weights.size(2) // 2
    mid2 = weights.size(3) // 2
    with torch.no_grad():
        weights[:, :, mid1, mid2] = q[: weights.size(0), : weights.size(1)]
        weights.mul_(gain)

def patchify_inputs(x, patch_size, overlap):
    x = (
        x.unfold(2, patch_size, patch_size // overlap)
            .unfold(3, patch_size, patch_size // overlap)
            .permute(0, 2, 3, 1, 4, 5) #b, p_x, p_y, c, x, y
    )
    n_patches_x = x.shape[1]
    n_patches_y = x.shape[2]
    x = x.reshape(
        x.shape[0] * x.shape[1] * x.shape[2], x.shape[3], x.shape[4], x.shape[5]
    )
    return x, n_patches_x, n_patches_y