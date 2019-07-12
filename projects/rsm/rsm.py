import matplotlib.pyplot as plt
import math

import torch
from torch import nn

from active_dendrites import ActiveDendriteLayer
from util import get_grad_printer, activity_square
from nupic.torch.modules.k_winners import KWinners


def topk_mask(x, k=2):
    """
    Simple functional version of KWinnersMask/KWinners since
    autograd function apparently not currently exportable by JIT
    """
    res = torch.zeros_like(x)
    topk, indices = x.topk(k, sorted=False)
    return res.scatter(-1, indices, 1)


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
        Receive input as hidden memory state from RSM, batch
        x^B is with shape (batch_size, total_cells)

        Output is (batch_size, d_out)
        """
        x = self.layers(x)
        return x.view(-1, self.d_out)


class RSMLayer(torch.nn.Module):
    def __init__(self, d_in=28 * 28, d_out=28 * 28, m=200, n=6, k=25,
                 k_winner_cells=1, gamma=0.5, eps=0.5, activation_fn='tanh',
                 cell_winner_softmax=False, active_dendrites=None,
                 col_output_cells=None, embed_dim=0, vocab_size=0, 
                 bsz=64, dropout_p=0.0, decode_from_full_memory=False,
                 debug_log_names=None, mask_shifted_pi=False, do_inhibition=True,
                 boost_strat='rsm_inhibition', pred_gain=1.0, x_b_norm=False,
                 boost_strength=1.0, mult_integration=False, boost_strength_factor=1.0,
                 debug=False, visual_debug=False, use_bias=True,
                 **kwargs):
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
        self.dropout_p = dropout_p

        self.total_cells = m * n
        self.bsz = bsz

        # Buffers / intermediates
        self.y = torch.zeros((self.bsz, self.total_cells), requires_grad=True)
        self.sigma = torch.zeros((self.bsz, self.total_cells), requires_grad=True)

        # Tweaks
        self.activation_fn = activation_fn
        self.active_dendrites = active_dendrites
        self.col_output_cells = col_output_cells
        self.cell_winner_softmax = cell_winner_softmax
        self.decode_from_full_memory = decode_from_full_memory
        self.boost_strat = boost_strat
        self.pred_gain = pred_gain
        self.x_b_norm = x_b_norm
        self.mask_shifted_pi = mask_shifted_pi
        self.do_inhibition = do_inhibition
        self.boost_strength = boost_strength
        self.mult_integration = mult_integration

        self.debug = debug
        self.visual_debug = visual_debug
        self.debug_log_names = debug_log_names

        self.dropout = nn.Dropout(p=self.dropout_p)

        self.kwinners_col = KWinners(self.m, self.k / self.m, 
                                     boost_strength=self.boost_strength,
                                     duty_cycle_period=5000,
                                     boost_strength_factor=boost_strength_factor)
        self.linear_a = nn.Linear(d_in, m, bias=use_bias)  # Input weights (shared per group / proximal)
        if self.active_dendrites:
            self.linear_b = ActiveDendriteLayer(self.total_cells, self.total_cells, 
                                                n_dendrites=self.active_dendrites)
        else:
            self.linear_b = nn.Linear(self.total_cells, self.total_cells, bias=use_bias)  # Recurrent weights (per cell)

        decode_d_in = self.total_cells if self.decode_from_full_memory else m
        self.linear_d = nn.Linear(decode_d_in, d_out, bias=use_bias)  # Decoding through bottleneck

    def _debug_log(self, tensor_dict, truncate_len=400):
        if self.debug:
            for name, t in tensor_dict.items():
                if not self.debug_log_names or name in self.debug_log_names:
                    _type = type(t)
                    if _type in [int, float, bool]:
                        size = '-'
                    else:
                        size = t.size()
                        _type = t.dtype
                        if t.numel() > truncate_len:
                            t = "..truncated.."
                    print([name, t, size, _type])
        if self.visual_debug:
            for name, t in tensor_dict.items():
                if not self.debug_log_names or name in self.debug_log_names:
                    if isinstance(t, torch.Tensor):
                        t = t.detach().squeeze()
                        if t.dim() == 1:
                            t = t.flatten()
                            size = t.numel()
                            is_cell_level = t.numel() == self.total_cells and self.n > 1
                            if is_cell_level:
                                plt.imshow(t.view(self.m, self.n).t(), origin='bottom', extent=(0, self.m-1, 0, self.n))
                            else:
                                plt.imshow(activity_square(t))
                            tmin = t.min()
                            tmax = t.max()
                            tsum = t.sum()
                            plt.title("%s (%s, rng: %.3f-%.3f, sum: %.3f)" % (name, size, tmin, tmax, tsum))
                            plt.show()

    def _post_epoch(self, epoch):
        self.kwinners_col.update_boost_strength()

    def _register_hooks(self):
        """Utility function to call retain_grad and Pytorch's register_hook
        in a single line
        """
        for label, t in [
                ('y', self.y), 
                ('sigma', self.sigma), 
                ('linear_b', self.linear_b.weight),
        ]:
            t.retain_grad()
            t.register_hook(get_grad_printer(label))

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
        # Col activation from inputs repeated for each cell
        z_a = self.linear_a(x_a).repeat_interleave(self.n, 1)
        # Cell activation from recurrent input (dropped out)
        z_b = self.linear_b(self.dropout(x_b)).view(bsz, self.total_cells)
        self._debug_log({'z_a': z_a, 'z_b': z_b})
        if self.mult_integration:
            self.sigma = z_a * z_b
        else:
            self.sigma = z_a + z_b
        return self.sigma  # total_cells

    def _k_winners(self, sigma, pi, bsz):
        # Group-wise max pooling
        lambda_ = self._group_max(pi)

        # Cell-level mask: Make a bsz x total_cells binary mask of top 1 cell in each column
        if self.n == self.k_winner_cells:
            # Usually just in n=1 case, no need to choose winners
            M_pi = torch.ones(bsz, self.total_cells, device=sigma.device)
        else:
            mask = topk_mask(pi.view(bsz * self.m, self.n), self.k_winner_cells)
            M_pi = mask.view(bsz, self.total_cells).detach()

        if self.boost_strat == 'rsm_inhibition':
            # Standard RSM-style inhibition via phi matrix

            self._debug_log({'lambda_': lambda_})

            # Column-level mask: Make a bsz x total_cells binary mask of top self.k columns
            mask = topk_mask(lambda_, self.k)
            M_lambda = mask.view(bsz, self.m, 1).repeat(1, 1, self.n).view(bsz, self.total_cells).detach()

            self._debug_log({'M_pi': M_pi, 'M_lambda': M_lambda})

            y_pre_act = M_pi * M_lambda * self.sigma

            del M_lambda

        elif self.boost_strat == 'col_boosting':
            # Experiment with HTM style boosted k-winner (TODO: should we prevent BP through masks like RSM?)
            winning_columns = self.kwinners_col(lambda_).view(bsz, self.m, 1).repeat(1, 1, self.n).view(bsz, self.total_cells)
            self._debug_log({'winning_columns': winning_columns})
            premask_act = pi if self.mask_shifted_pi else self.sigma
            y_pre_act = M_pi * winning_columns * premask_act

        del M_pi

        return y_pre_act

    def _inhibited_masking_and_prediction(self, sigma, phi, bsz):
        """
        Compute y_lambda
        """
        # Apply inhibition to non-neg shifted sigma
        inh = (1-phi) if self.do_inhibition else 1
        pi = inh * (self.sigma - self.sigma.min() + 1)
        self._debug_log({'pi': pi})

        pi = pi.detach()  # Prevent gradients from flowing through inhibition/masking

        if self.col_output_cells:
            max_val = pi.max()
            # Clamp x fixed output cells per group/col 
            # Last x cells in each col
            pi[:, -self.col_output_cells:] = max_val

        y_pre_act = self._k_winners(self.sigma, pi, bsz)

        # Mask-based sparsening
        activation = {
            'tanh': torch.tanh,
            'relu': nn.functional.relu
        }[self.activation_fn]
        self.y = activation(y_pre_act)  # 1 x total_cells

        # Decode prediction through group-wise max bottleneck
        decode_input = self.y if self.decode_from_full_memory else self._group_max(self.y)
        output = self.linear_d(decode_input)

        return (self.y, output)

    def _update_memory_and_inhibition(self, y, phi, psi, bsz):
        # Get updated psi (memory state), decay inactive
        psi = torch.max(psi * self.eps, y)

        # Update phi for next step (decay inhibition cells)
        phi = torch.max(phi * self.gamma, y)

        return (phi, psi)

    def forward(self, x_a_batch, hidden):
        """
        :param x_a_batch: Input batch of batch_size items from generating process (batch_size, d_in)
        """
        bsz = x_a_batch.size(0)

        x_b, phi, psi = hidden
        self._debug_log({'x_b': x_b})

        self._debug_log({'x_a_batch': x_a_batch})

        self.sigma = self._fc_weighted_ave(x_a_batch, x_b, bsz)
        self._debug_log({'sigma': self.sigma})

        self.y, pred_output = self._inhibited_masking_and_prediction(self.sigma, phi, bsz)
        self._debug_log({'y': self.y, 'pred_output': pred_output})

        phi, psi = self._update_memory_and_inhibition(self.y, phi, psi, bsz)
        self._debug_log({'phi': phi, 'psi': psi})

        # Update recurrent input / output x_b
        if self.x_b_norm:
            # Normalizing scalar (force sum(x_b) == 1), avoiding div 0... this small enough?
            alpha = (psi.sum(dim=1) + 1e-9).unsqueeze(dim=1)
            x_b = self.pred_gain * psi / alpha
        else:
            x_b = self.pred_gain * psi

        hidden = (x_b, phi, psi)
        return (pred_output, hidden)


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
