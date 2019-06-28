
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import torch


def activity_square(vector):
    n = len(vector)
    side = int(np.sqrt(n))
    if side ** 2 < n:
        side += 1
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
    Plot column activations for each combination of input and actual next input
    Should show mini-column union activity (subsets of column-level activity
    which predict next input) in the RSM model.
    """
    fig, axs = plt.subplots(n_labels, n_labels, dpi=200, 
                            sharex=True, sharey=True,
                            gridspec_kw={'wspace': 0, 'hspace': 0})
    for i in range(n_labels):
        for j in range(n_labels):
            ax = axs[i][j]
            ax.axis('off')
            key = '%d-%d' % (i, j)
            if key in distrs:
                activity_arr = distrs[key]
                ax.set_title("%d -> %d" % (i, j), fontsize=7)
                dist = torch.stack(activity_arr)
                mean_act = dist.mean(dim=0)
                ax.imshow(activity_square(mean_act))
            else:
                ax.set_axis_off()

    return fig


def plot_activity(distrs, n_labels=10, level='column'):
    """
    Plot column activations for each combination of input and actual next input
    Should show mini-column union activity (subsets of column-level activity
    which predict next input) in the RSM model.
    """
    n_plots = len(distrs.keys())
    fig, axs = plt.subplots(n_plots, 1, dpi=144, gridspec_kw={'hspace': 0.7})
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
                col_act = dist.max(dim=2).values
                if level == 'column':
                    act = col_act
                elif level == 'cell':
                    col = col_act.view(bsz, m, 1)
                    act = torch.cat((dist, col), 2).view(bsz, m, n + 1)
                mean_act = act.mean(dim=0).cpu()
                ax.imshow(mean_act.t(), origin='bottom', extent=(0, m-1, 0, n+1))
                ax.plot([0, m-1], [n, n], linewidth=1)
                ax.axis('off')
                ax.set_title(key, fontsize=5)
    return fig
