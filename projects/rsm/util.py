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

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from torch.nn.functional import cosine_similarity


def square_size(n):
    side = int(np.sqrt(n))
    if side ** 2 < n:
        side += 1
    return side


def activity_square(vector):
    n = len(vector)
    side = square_size(n)
    square = torch.zeros(side ** 2)
    square[:n] = vector
    return square.view(side, side)


def fig2img(fig):
    canvas = FigureCanvas(fig)
    canvas.draw()

    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype="uint8").reshape(
        int(height), int(width), 3
    )
    return img


def plot_confusion_matrix(
    y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues
):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig = Figure()
    ax = fig.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    return ax, fig


def plot_activity_grid(distrs, n_labels=10):
    """
    For flattened models, plot cell activations for each combination of
    input and actual next input
    """
    fig, axs = plt.subplots(
        n_labels,
        n_labels,
        dpi=300,
        gridspec_kw={"hspace": 0.7, "wspace": 0.7},
        sharex=True,
        sharey=True,
    )
    for i in range(n_labels):
        for j in range(n_labels):
            key = "%d-%d" % (i, j)
            if key in distrs:
                activity_arr = distrs[key]
                dist = torch.stack(activity_arr)
                ax = axs[i][j]
                mean_act = activity_square(dist.mean(dim=0).cpu())
                side = mean_act.size(0)
                ax.imshow(mean_act, origin="bottom", extent=(0, side, 0, side))
            else:
                ax.set_visible(False)
            ax.axis("off")
            ax.set_title(key, fontsize=5)
    return fig


def plot_activity(distrs, n_labels=10, level="column"):
    """
    Plot column activations for each combination of input and actual next input
    Should show mini-column union activity (subsets of column-level activity
    which predict next input) in the RSM model.
    """
    n_plots = len(distrs.keys())
    fig, axs = plt.subplots(n_plots, 1, dpi=300, gridspec_kw={"hspace": 0.7})
    pi = 0
    for i in range(n_labels):
        for j in range(n_labels):
            key = "%d-%d" % (i, j)
            if key in distrs:
                activity_arr = distrs[key]
                dist = torch.stack(activity_arr)
                ax = axs[pi]
                pi += 1
                bsz, m, n = dist.size()
                no_columns = n == 1
                col_act = dist.max(dim=2).values
                if level == "column" or no_columns:
                    act = col_act
                elif level == "cell":
                    col = col_act.view(bsz, m, 1)
                    act = torch.cat((dist, col), 2).view(bsz, m, n + 1)
                mean_act = act.mean(dim=0).cpu()
                if no_columns:
                    mean_act = activity_square(mean_act)
                    side = mean_act.size(0)
                    ax.imshow(mean_act, origin="bottom", extent=(0, side, 0, side))
                else:
                    ax.imshow(
                        mean_act.t(), origin="bottom", extent=(0, m - 1, 0, n + 1)
                    )
                    ax.plot([0, m - 1], [n, n], linewidth=0.4)
                ax.axis("off")
                ax.set_title(key, fontsize=5)
    return fig


def _repr_similarity_grid(
    ax,
    activity_arr,
    cmap=plt.cm.Blues,
    normalize=False,
    labels=None,
    title=None,
    tick_fontsize=2,
    fontsize=1.2,
):
    n_labels = len(labels)
    grid = torch.zeros(n_labels, n_labels)

    # Compute grid (cosine similarity)
    for i, act1 in enumerate(activity_arr):
        for j, act2 in enumerate(activity_arr):
            if j > i:
                break
            if act1 is not None and act2 is not None:
                sim = cosine_similarity(act1, act2, dim=0)
                grid[i, j] = grid[j, i] = sim

    ax.imshow(grid, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(grid.shape[1]),
        yticks=np.arange(grid.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=labels,
        yticklabels=labels,
        title=title,
    )
    ax.tick_params(labelsize=tick_fontsize)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = grid.max() / 2.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j,
                i,
                format(grid[i, j], ".2f"),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if grid[i, j] > thresh else "black",
            )


def plot_representation_similarity(
    distrs, n_labels=10, title=None, save=None, fontsize=1.6
):
    """
    Plot grid showing representation similarity between distributions passed
    into distrs dict.
    """
    fig, axs = plt.subplots(1, 2, dpi=300)
    ax_id = 0

    col_activities = []
    cell_activities = []
    labels = []

    for i in range(n_labels):
        for j in range(n_labels):
            key = "%d-%d" % (i, j)
            col_act = cell_act = None
            if key in distrs:
                activity_arr = distrs[key]
                dist = torch.stack(activity_arr)
                ax_id += 1
                size = dist.size()
                if len(size) == 3:
                    bsz, m, n = size
                    tc = m * n
                else:
                    bsz, m = size
                    tc = m

                if m != tc:
                    col_act = (
                        dist.max(dim=-1).values.view(bsz, m).mean(dim=0).flatten().cpu()
                    )
                    col_activities.append(col_act)
                # TODO: Check reshaping here
                cell_act = dist.view(bsz, tc).mean(dim=0).flatten().cpu()

                labels.append(key)
                cell_activities.append(cell_act)

    if col_activities:
        _repr_similarity_grid(
            axs[0], col_activities, labels=labels, title="Column", fontsize=fontsize
        )
    _repr_similarity_grid(
        axs[1], cell_activities, labels=labels, title="Cell", fontsize=fontsize
    )
    suptitle = "Repr Similarity (Cos)"
    if title:
        suptitle += " - " + title
    fig.suptitle(suptitle)
    if save:
        fig.savefig(save)
    return fig


def get_grad_printer(msg):
    """
    This function returns a printer function, that prints information about a
    tensor's gradient. Used by register_hook in the backward pass.
    """
    def printer(grad):
        if grad.nelement() == 1:
            print(f"{msg} {grad}")
        else:
            print(
                f"{msg} shape: {grad.shape}"
                f" {len(grad.nonzero())}/{grad.numel()} nonzero"
                f" max: {grad.max()} min: {grad.min()}"
                f" mean: {grad.mean()}"
            )

    return printer


def count_parameters(model, exclude=None):
    params = 0
    for n, p in model.named_parameters():
        if p.requires_grad and (exclude is None or exclude not in n):
            params += p.numel()
    return params


def print_epoch_values(ret):
    """
    Print dictionary of epoch values with large arrays removed
    """
    print_ret = {}
    for key, _val in ret.items():
        if not key.startswith("img_") and not key.startswith("hist_"):
            print_ret[key] = ret[key]
    return print_ret


def _plot_grad_flow(model, top=0.01):
    """
    Plots the gradients flowing through different layers in the net during
    training. Can be used for checking for possible gradient
    vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            zg = False
            if p.grad is not None:
                pmax = p.grad.abs().max()
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(pmax)
                zg = pmax == 0
            else:
                ave_grads.append(0)
                max_grads.append(0)
                zg = True
            if zg:
                n += " *"
            layers.append(n)
    print("Gradients", max_grads)
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=top)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow (* indicates 0 grad)")
    plt.grid(True)
    LABELS = ["max-gradient", "mean-gradient", "zero-gradient"]
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
        ],
        LABELS
    )


def plot_cluster_weights(model):
    # Switched to standard PCA
    # To identify column formation we'll need to combine weights
    # linear_a_int/linear_b_int since clusters may include cells
    # across FF and predictive partitions
    pca_3d = PCA(n_components=3)
    w_a = model.linear_a.weight.data.cpu()
    w_b = model.linear_b.weight.data.cpu()
    fig, axs = plt.subplots(1, 2, dpi=200)

    w_a_emb = pca_3d.fit_transform(w_a)
    axs[0].scatter(w_a_emb[:, 0], w_a_emb[:, 1], c=w_a_emb[:, 2], s=1.5, alpha=0.6)
    axs[0].set_title("FF input - %d cells" % w_a.shape[0])

    if len(w_b):
        w_b_emb = pca_3d.fit_transform(w_b)
        axs[1].scatter(w_b_emb[:, 0], w_b_emb[:, 1], c=w_b_emb[:, 2], s=1.5, alpha=0.6)
        axs[1].set_title("Rec input - %d cells" % w_b.shape[0])
    return fig


def print_aligned_sentences(s1, s2, labels=None):
    widths = []
    s1 = s1.split()
    s2 = s2.split()
    for w1, w2 in zip(s1, s2):
        widths.append(max([len(w1), len(w2)]))
    out1 = out2 = ""
    for w1, w2, width in zip(s1, s2, widths):
        out1 += w1.ljust(width + 1)
        out2 += w2.ljust(width + 1)
    print("%s: %s" % (labels[0] if labels else "s1", out1))
    print("%s: %s" % (labels[1] if labels else "s2", out2))


def _is_long(x):
    if hasattr(x, 'data'):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)

def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output

def smoothed_cross_entropy(inputs, target, weight=None, ignore_index=-100, reduction='mean',
                  smooth_eps=None, smooth_dist=None, from_logits=True):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-liklihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1. - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())

    return loss


class SmoothedCrossEntropyLoss(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean', smooth_eps=None, smooth_dist=None, from_logits=True):
        super(SmoothedCrossEntropyLoss, self).__init__(weight=weight,
                                               ignore_index=ignore_index, reduction=reduction)
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return smoothed_cross_entropy(input, target, weight=self.weight, ignore_index=self.ignore_index,
                             reduction=self.reduction, smooth_eps=self.smooth_eps,
                             smooth_dist=smooth_dist, from_logits=self.from_logits)

def plot_tensors(model, tuples, detailed=False, return_fig=False):
    """
    Plot first item in batch across multiple layers
    """
    n_tensors = len(tuples)
    fig, axs = plt.subplots(model.n_layers, n_tensors, dpi=144)
    for i, (label, val) in enumerate(tuples):
        for l in range(model.n_layers):
            layer_idx = model.n_layers - l - 1
            ax = axs[layer_idx][i] if n_tensors > 1 else axs[layer_idx]
            # Get layer's values (from either list or tensor)
            # Outputs can't be stored in tensors since dimension heterogeneous
            if isinstance(val, list):
                if val[l] is None:
                    ax.set_visible(False)
                    t = None
                else:
                    t = val[l].detach()[0]
            else:
                t = val.detach()[l, 0]
            mod = list(model.children())[l]
            if t is not None:
                size = t.numel()
                is_cell_level = t.numel() == mod.total_cells and mod.n > 1
                if is_cell_level:
                    ax.imshow(
                        t.view(mod.m, mod.n).t(),
                        origin="bottom",
                        extent=(0, mod.m - 1, 0, mod.n + 1),
                    )
                else:
                    ax.imshow(activity_square(t))
                tmin = t.min()
                tmax = t.max()
                tsum = t.sum()
                title = "L%d %s" % (l+1, label)
                if detailed:
                    title += " (%s, rng: %.3f-%.3f, sum: %.3f)" % (size, tmin, tmax, tsum)
                ax.set_title(title)
    if return_fig:
        return fig
    else:
        plt.show()