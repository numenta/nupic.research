from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F


def topk_mask(a, k, dim=0, softmax=False):
    """
    Return a 1 for the top k elements in the last dim of a, 0 otherwise
    """
    values, indices = torch.topk(a, k, dim=dim)
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


class RSMPredictor(torch.nn.Module):
    def __init__(self, d_in=28 * 28, d_out=10, hidden_size=20):
        """

        """
        super(RSMPredictor, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.hidden_size = hidden_size

        self.layers = nn.Sequential(
            nn.Linear(self.d_in, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.d_out),
            nn.LeakyReLU()
        )

    def forward(self, x):
        """
        Receive input as hidden memory state from RSM, batched
        x is with shape (seq_len, batch_size, total_cells)

        Output is (seq_len * batch_size, d_out)
        """
        sl, bsz, _ = x.size()
        x = self.layers(x)
        return x.view(sl * bsz, self.d_out)


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
        :param activity: activity vector (bsz x total_cells)

        Returns max cell activity in each group
        """
        return activity.view(activity.size(0), self.m, self.n).max(dim=2).values

    def _fc_weighted_ave(self, x_a, x_b, bsz):
        """
        Compute sigma (weighted sum for each cell j in group i (mxn))
        """
        z_a = self.linear_a(x_a).repeat(1, self.n)  # z_a (repeated for each cell)
        z_b = self.linear_b(x_b).view(bsz, self.total_cells)
        sigma = z_a + z_b
        self._debug_log({'z_a + z_b = sigma': sigma})
        return sigma  # total_cells x bsz

    def _inhibited_masking_and_prediction(self, sigma, phi, bsz):
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

        # Mask: most active cell in group
        M_pi = topk_mask(pi.view(bsz, self.m, self.n), self.k_winner_cells, dim=2, softmax=self.cell_winner_softmax)
        M_pi = M_pi.view(bsz, self.total_cells)

        # Mask: most active group (m)
        M_lambda = topk_mask(lambda_i, self.k, dim=1).view(bsz, self.m, 1).repeat(1, 1, self.n)
        M_lambda = M_lambda.view(bsz, self.total_cells)
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
        psi = torch.max(psi * self.eps, y)

        # Update phi for next step (decay inactive cells)
        phi = torch.max(phi * self.gamma, y)

        return (phi, psi)

    def init_hidden(self, batch_size):
        # TODO: Update with batch_size
        weight = next(self.parameters())
        x_b = weight.new_zeros((batch_size, self.total_cells), dtype=torch.float32, requires_grad=False)
        phi = weight.new_zeros((batch_size, self.total_cells), dtype=torch.float32, requires_grad=False)
        psi = weight.new_zeros((batch_size, self.total_cells), dtype=torch.float32, requires_grad=False)
        return (x_b, phi, psi)

    def forward(self, x_a_batch, hidden):
        """
        :param x_a_batch: Input batch of batch_size seq_len sequences (seq_len, batch_size, d_in)
        """
        seq_len = x_a_batch.size(0)
        bsz = x_a_batch.size(1)

        output = None
        x_bs = None

        x_b, phi, psi = hidden

        for seqi in range(seq_len):
            x_a_row = x_a_batch[seqi, :]

            self._debug_log({'seqi': seqi, 'x_a_row': x_a_row})

            sigma = self._fc_weighted_ave(x_a_row, x_b, bsz)
            self._debug_log({'sigma': sigma})

            y, x_a_next = self._inhibited_masking_and_prediction(sigma, phi, bsz)
            self._debug_log({'y': y, 'x_a_next': x_a_next})

            phi, psi = self._update_memory_and_inhibition(y, phi, psi)
            self._debug_log({'phi': phi, 'psi': psi})

            # Update recurrent input / output x_b
            alpha = psi.sum()
            if not alpha:
                alpha = 1.0
            x_b = (psi / alpha)  # Normalizing scalar (force sum(x_b) == 1)
            self._debug_log({'x_b': x_b})

            # Detach recurrent hidden layer to avoid 
            # "Trying to backward through the graph a second time" recursion error
            y = y.detach()
            sigma = sigma.detach()
            phi = phi.detach()
            psi = psi.detach()

            if output is None:
                output = x_a_next
            else:
                output = torch.cat((output, x_a_next))
            if x_bs is None:
                x_bs = x_b
            else:
                x_bs = torch.cat((x_bs, x_b))

        hidden = (x_b, phi, psi)

        return (output.view(seq_len, bsz, self.d_out), hidden, x_bs.view(seq_len, bsz, self.total_cells))


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
