
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
from torch.nn.functional import cosine_similarity
import math


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
                col_act = dist.max(dim=2).values
                if level == 'column':
                    act = col_act
                elif level == 'cell':
                    col = col_act.view(bsz, m, 1)
                    act = torch.cat((dist, col), 2).view(bsz, m, n + 1)
                mean_act = act.mean(dim=0).cpu()
                ax.imshow(mean_act.t(), origin='bottom', extent=(0, m-1, 0, n+1))
                ax.plot([0, m-1], [n, n], linewidth=0.4)
                ax.axis('off')
                ax.set_title(key, fontsize=5)
    return fig


def _repr_similarity_grid(ax, activity_arr, cmap=plt.cm.Blues, 
                          normalize=False, labels=None, title=None,
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
    ax.tick_params(labelsize=fontsize)

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


def plot_representation_similarity(distrs, n_labels=10, title=None, save=None, fontsize=1.4):
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
                bsz, m, n = dist.size()

                # Store averaged column and cell activity, and append to each list
                col_act = dist.max(dim=2).values.view(bsz, m).mean(dim=0).flatten().cpu()
                # TODO: Check reshaping here
                cell_act = dist.view(bsz, m * n).mean(dim=0).flatten().cpu()

                labels.append(key)
                col_activities.append(col_act)
                cell_activities.append(cell_act)

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


class AdamW(torch.optim.Optimizer):
    """Implements AdamW algorithm.
    It has been proposed in `Fixing Weight Decay Regularization in Adam`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    .. Fixing Weight Decay Regularization in Adam:
    https://arxiv.org/abs/1711.05101
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # according to the paper, this penalty should come after the bias correction
                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # w = w - wd * lr * w
                if group['weight_decay'] != 0:
                    p.data.add_(-group['weight_decay'] * group['lr'], p.data)

                # w = w - lr * w.grad
                p.data.addcdiv_(-step_size, exp_avg, denom)

                # w = w - wd * lr * w - lr * w.grad
                # See http://www.fast.ai/2018/07/02/adam-weight-decay/

        return loss