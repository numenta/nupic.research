from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


def topk_mask(a, b, dim=0, softmax=False):
    """
    Return a 1 for the top b elements in the last dim of a, 0 otherwise
    """
    values, indices = torch.topk(a, b)
    arr = a.new_zeros(a.size())  # Zeros, conserve device
    arr.scatter_(dim, indices, 1)
    return arr


class LocalLinear(nn.Module):
    """
    """
    def __init__(self, in_features, local_features, kernel_size, stride=1, bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        fold_num = (in_features - self.kernel_size) // self.stride + 1
        self.lc = nn.ModuleList([deepcopy(nn.Linear(kernel_size, local_features, bias=bias))
                                 for _ in range(fold_num)])

    def forward(self, x):
        x = x.unfold(-1, size=self.kernel_size, step=self.stride)
        fold_num = x.shape[1]
        x = torch.cat([self.lc[i](x[:, i, :]) for i in range(fold_num)], 1)
        return x


class ActiveDendriteLayer(torch.nn.Module):
    """
    Local layer for active dendrites. Similar to a non-shared weight version of a
    2D Conv layer.

    Note that dendrites are fully connected to input, local layer used only for connecting
    neurons and their dendrites
    """
    def __init__(self, input_dim, n_cells=50, n_dendrites=3):
        super(ActiveDendriteLayer, self).__init__()
        self.n_cells = n_cells
        self.n_dendrites = n_dendrites

        total_dendrites = n_dendrites * n_cells
        self.linear_dend = nn.Linear(input_dim, total_dendrites)
        self.linear_neuron = LocalLinear(total_dendrites, 1, n_dendrites, stride=n_dendrites)

    def __repr__(self):
        return "ActiveDendriteLayer neur=%d, dend per neuron=%d" % (self.n_cells, self.n_dendrites)

    def forward(self, x):
        x = F.relu(self.linear_dend(x))
        x = self.linear_neuron(x)
        return x


class RSMLayer(torch.nn.Module):
    def __init__(self, d_in=28 * 28, d_out=28 * 28, m=200, n=6, k=25,
                 k_winner_cells=1, gamma=0.5, eps=0.5, activation_fn='tanh',
                 cell_winner_softmax=False, active_dendrites=None,
                 col_output_cells=None,
                 embed_dim=0, vocab_size=0, debug=False, **kwargs):
        """
        RSM Layer as specified by Rawlinson et al 2019

        :param d_in: Dimension of input
        :param m: Number of groups
        :param n: Cells per group
        :param k: # of groups to win in topk() (sparsity)
        :param gamma: Inhibition decay rate (0-1)
        :param eps: Integrated encoding decay rate (0-1)

        """
        super(RSMLayer, self).__init__()
        self.k = k
        self.k_winner_cells = k_winner_cells
        self.m = m
        self.n = n
        self.gamma = gamma
        self.eps = eps
        self.d_in = d_in
        self.d_out = d_out

        # Tweaks
        self.activation_fn = activation_fn
        self.active_dendrites = active_dendrites
        self.col_output_cells = col_output_cells

        self.cell_winner_softmax = cell_winner_softmax

        self.total_cells = m * n
        self.debug = debug

        self.embedding = None
        if embed_dim:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            initrange = 0.1
            self.embedding.weight.data.uniform_(-initrange, initrange)

        self.linear_a = nn.Linear(d_in, m)  # Input weights (shared per group / proximal)
        if self.active_dendrites:
            self.linear_b = ActiveDendriteLayer(self.total_cells, self.total_cells, 
                                                n_dendrites=self.active_dendrites)
        else:
            self.linear_b = nn.Linear(self.total_cells, self.total_cells)  # Recurrent weights (per cell)
        self.linear_d = nn.Linear(m, d_out)  # Decoding through bottleneck

    def _debug_log(self, tensor_dict):
        if self.debug:
            for name, t in tensor_dict.items():
                _type = type(t)
                if _type in [int, float, bool]:
                    size = '-'
                else:
                    size = t.size()
                    _type = t.dtype
                print([name, t, size, _type])

    def _track_weights(self):
        ret = {}
        ret['hist_w_a'] = self.linear_a.weight.cpu()
        ret['hist_w_b'] = self.linear_b.weight.cpu()
        ret['hist_w_d'] = self.linear_d.weight.cpu()
        return ret

    def _group_max(self, activity):
        """
        :param activity: 1D vector of activity (m x n)

        Returns max cell activity in each group
        """
        return activity.view(self.m, self.n).max(dim=1).values

    def _fc_weighted_ave(self, x, x_b):
        """
        Compute sigma (weighted sum for each cell j in group i (mxn))
        """
        z_a = self.linear_a(x).view(self.m, 1).repeat(1, self.n)  # z_a (repeated for each cell, mxn)
        self._debug_log({'z_a': z_a})
        z_b = self.linear_b(x_b).view(self.m, self.n)  # z_b (sigma now dim mxn)
        sigma = z_a + z_b
        self._debug_log({'z_a + z_b = sigma': sigma})
        return sigma

    def _inhibited_masking_and_prediction(self, sigma, phi):
        """
        Compute y_lambda
        """
        # Apply inhibition and shift to be non-neg
        pi = (1 - phi) * (sigma - sigma.min() + 1)
        self._debug_log({'pi': pi})

        if self.col_output_cells:
            max_val = pi.max()
            # Clamp x fixed output cells per group/col 
            # Last x cells in each col
            pi[:, -self.col_output_cells:] = max_val

        # Group-wise max pooling
        lambda_i = self._group_max(pi)
        self._debug_log({'lambda_i': lambda_i})

        # Mask: most active cell in group (total_cells)
        M_pi = topk_mask(pi, self.k_winner_cells, dim=1, softmax=self.cell_winner_softmax)

        # Mask: most active group (m)
        M_lambda = topk_mask(lambda_i, self.k, dim=0).view(self.m, 1).repeat(1, self.n)
        self._debug_log({'M_pi': M_pi, 'M_lambda': M_lambda})

        # Mask-based sparsening
        activation = {
            'tanh': torch.tanh,
            'relu': nn.functional.relu
        }[self.activation_fn]
        y = activation(M_pi * M_lambda * sigma)  # 1 x total_cells

        # Decode prediction through group-wise max bottleneck
        x_a_pred = self.linear_d(self._group_max(y))

        del M_pi
        del M_lambda

        return (y, x_a_pred)

    def _update_memory_and_inhibition(self, y, phi, psi):
        # Get updated psi (memory state), decay inactive inactive
        # psi *= self.eps
        psi = torch.max(psi * self.eps, y)

        # Update phi for next step (decay inactive cells)
        # phi *= self.gamma
        phi = torch.max(phi * self.gamma, y)

        return (phi, psi)

    def init_hidden(self, batch_size):
        # TODO: Update with batch_size
        weight = next(self.parameters())
        x_b = weight.new_zeros(self.total_cells, dtype=torch.float32, requires_grad=False)
        phi = weight.new_zeros((self.m, self.n), dtype=torch.float32, requires_grad=False)
        psi = weight.new_zeros((self.m, self.n), dtype=torch.float32, requires_grad=False)
        return (x_b, phi, psi)

    def forward(self, batch_x, hidden):
        """
        :param x: Input batch (batch_size, d_in, or Tensor of word ids if using embedding)

        TODO: Update to take batch_x of dim (seq_len, batch, input_size), parallel to LSTM
        """
        # Can we vectorize across multiple batches?
        batch_size = batch_x.size(0)
        x_a_nexts = []
        batch_x.new_zeros((batch_size, self.d_in))  # Conserve device
        x_b, phi, psi = hidden
        x_bs = []

        for i, x in enumerate(batch_x):
            self._debug_log({'i': i, 'x': x})

            if self.embedding:
                # Embedded word vector
                x = self.embedding(x)

            sigma = self._fc_weighted_ave(x.flatten(), x_b)
            self._debug_log({'sigma': sigma})

            y, x_a_next = self._inhibited_masking_and_prediction(sigma, phi)
            self._debug_log({'y': y, 'x_a_next': x_a_next})

            phi, psi = self._update_memory_and_inhibition(y, phi, psi)
            self._debug_log({'phi': phi, 'psi': psi})

            # Update recurrent input / output x_b
            alpha = psi.sum()
            if not alpha:
                alpha = 1.0
            x_b = (psi / alpha).flatten()  # Normalizing scalar (force sum(x_b) == 1)
            self._debug_log({'x_b': x_b})

            # Detach recurrent hidden layer to avoid 
            # "Trying to backward through the graph a second time" recursion error
            y = y.detach()
            sigma = sigma.detach()
            phi = phi.detach()
            psi = psi.detach()

            x_a_nexts.append(x_a_next)
            x_bs.append(x_b)

        hidden = (x_b, phi, psi)
        return (torch.stack(x_a_nexts), hidden, torch.stack(x_bs))


if __name__ == "__main__":
    batch_size, d_in = 50, 64

    x = torch.randn(batch_size, d_in)
    y = torch.randn(batch_size, d_in)

    model = RSMLayer(d_in)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(500):
        y_pred = model(x)

        loss = criterion(y_pred, y)
        print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
