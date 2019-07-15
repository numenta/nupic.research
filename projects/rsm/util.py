
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
from torch.nn.functional import cosine_similarity
import math


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
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return img


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true.cpu(), y_pred.cpu())

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = Figure()
    ax = fig.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    return ax, fig


def plot_activity_grid(distrs, n_labels=10):
    """
    For flattened models, plot cell activations for each combination of input and actual next input
    """
    fig, axs = plt.subplots(n_labels, n_labels, dpi=300, 
                            gridspec_kw={'hspace': 0.7, 'wspace': 0.7},
                            sharex=True, sharey=True)
    for i in range(n_labels):
        for j in range(n_labels):
            key = '%d-%d' % (i, j)
            if key in distrs:
                activity_arr = distrs[key]
                dist = torch.stack(activity_arr)
                ax = axs[i][j]
                mean_act = activity_square(dist.mean(dim=0).cpu())
                side = mean_act.size(0)
                ax.imshow(mean_act, origin='bottom', extent=(0, side, 0, side))
            else:
                ax.set_visible(False)
            ax.axis('off')
            ax.set_title(key, fontsize=5)
    return fig


def plot_activity(distrs, n_labels=10, level='column'):
    """
    Plot column activations for each combination of input and actual next input
    Should show mini-column union activity (subsets of column-level activity
    which predict next input) in the RSM model.
    """
    n_plots = len(distrs.keys())
    fig, axs = plt.subplots(n_plots, 1, dpi=300, gridspec_kw={'hspace': 0.7})
    pi = 0
    for i in range(n_labels):
        for j in range(n_labels):
            key = '%d-%d' % (i, j)
            if key in distrs:
                activity_arr = distrs[key]
                dist = torch.stack(activity_arr)
                ax = axs[pi]
                pi += 1
                bsz, m, n = dist.size()
                no_columns = n == 1
                col_act = dist.max(dim=2).values
                if level == 'column' or no_columns:
                    act = col_act
                elif level == 'cell':
                    col = col_act.view(bsz, m, 1)
                    act = torch.cat((dist, col), 2).view(bsz, m, n + 1)
                mean_act = act.mean(dim=0).cpu()
                if no_columns:
                    mean_act = activity_square(mean_act)
                    side = mean_act.size(0)
                    ax.imshow(mean_act, origin='bottom', extent=(0, side, 0, side))
                else:
                    ax.imshow(mean_act.t(), origin='bottom', extent=(0, m-1, 0, n+1))
                    ax.plot([0, m-1], [n, n], linewidth=0.4)
                ax.axis('off')
                ax.set_title(key, fontsize=5)
    return fig


def _repr_similarity_grid(ax, activity_arr, cmap=plt.cm.Blues, 
                          normalize=False, labels=None, title=None,
                          tick_fontsize=2,
                          fontsize=1.2):
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

    ax.imshow(grid, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    # ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(grid.shape[1]),
           yticks=np.arange(grid.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=labels, yticklabels=labels,
           title=title)
    ax.tick_params(labelsize=tick_fontsize)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = grid.max() / 2.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, format(grid[i, j], '.2f'),
                    ha="center", va="center",
                    fontsize=fontsize,
                    color="white" if grid[i, j] > thresh else "black")


def plot_representation_similarity(distrs, n_labels=10, title=None, save=None, fontsize=1.6):
    '''
    Plot grid showing representation similarity between distributions passed
    into distrs dict. 
    '''
    fig, axs = plt.subplots(1, 2, dpi=300)
    ax_id = 0

    col_activities = []
    cell_activities = []
    labels = []

    for i in range(n_labels):
        for j in range(n_labels):
            key = '%d-%d' % (i, j)
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
                    col_act = dist.max(dim=-1).values.view(bsz, m).mean(dim=0).flatten().cpu()
                    col_activities.append(col_act)
                # TODO: Check reshaping here
                cell_act = dist.view(bsz, tc).mean(dim=0).flatten().cpu()

                labels.append(key)
                cell_activities.append(cell_act)

    if col_activities:
        _repr_similarity_grid(axs[0], col_activities, labels=labels, title="Column", fontsize=fontsize)
    _repr_similarity_grid(axs[1], cell_activities, labels=labels, title="Cell", fontsize=fontsize)
    suptitle = "Repr Similarity (Cos)"
    if title:
        suptitle += " - " + title
    fig.suptitle(suptitle)
    if save:
        fig.savefig(save)
    return fig


def get_grad_printer(msg):
    """This function returns a printer function, that prints information about a  tensor's
    gradient. Used by register_hook in the backward pass.
    """
    def printer(grad):
        if grad.nelement() == 1:
            print(f"{msg} {grad}")
        else:
            print(f"{msg} shape: {grad.shape}"
                  f" {len(grad.nonzero())}/{grad.numel()} nonzero"
                  f" max: {grad.max()} min: {grad.min()}"
                  f" mean: {grad.mean()}")
    return printer


def count_parameters(model, exclude=None):
    params = 0
    for n, p in model.named_parameters():
        if p.requires_grad and (exclude is None or exclude not in n):
            params += p.numel()
    return params


def print_epoch_values(ret):
    '''
    Print dictionary of epoch values with large arrays removed
    '''
    print_ret = {}
    for key, val in ret.items():
        if not key.startswith('img_') and not key.startswith('hist_'):
            print_ret[key] = ret[key]
    return print_ret

