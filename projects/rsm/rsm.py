import torch
from torch import nn


def topk_mask(a, b, dim=0):
    """
    Return a 1 for the top b elements in the last dim of a, 0 otherwise
    """
    values, indices = torch.topk(a, b)
    arr = a.new_zeros(a.size())  # Zeros, conserve device
    arr.scatter_(dim, indices, 1)
    return arr


class RSMLayer(torch.nn.Module):
    def __init__(self, D_in=28 * 28, m=200, n=6, k=25, gamma=0.5, eps=0.5, debug=False):
        """
        RSM Layer as specified by Rawlinson et al 2019

        :param D_in: Dimension of input
        :param m: Number of groups
        :param n: Cells per group
        :param k: # of groups to win in topk() (sparsity)
        :param gamma: Inhibition decay rate (0-1)
        :param eps: Integrated encoding decay rate (0-1)
        """
        super(RSMLayer, self).__init__()
        self.k = k
        self.m = m
        self.n = n
        self.gamma = gamma
        self.eps = eps
        self.D_in = D_in

        self.total_cells = m * n
        self.debug = debug

        self.linear_a = nn.Linear(D_in, m)  # Input weights (shared per group / proximal)
        self.linear_b = nn.Linear(self.total_cells, self.total_cells)  # Recurrent weights (per cell)
        self.linear_d = nn.Linear(m, D_in)  # Decoding through bottleneck

        # self.register_buffer('z_a', torch.zeros(m, requires_grad=False))
        # self.register_buffer('z_b', torch.zeros(self.total_cells, requires_grad=False))
        # self.register_buffer('pi', torch.zeros(self.total_cells, requires_grad=False))
        # self.register_buffer('lambda_i', torch.zeros(self.total_cells, requires_grad=False))
        # self.register_buffer('M_pi', torch.zeros(self.total_cells, requires_grad=False))
        # self.register_buffer('M_lambda', torch.zeros(self.total_cells, requires_grad=False))
        # self.register_buffer('sigma', torch.zeros(self.total_cells, requires_grad=False))
        # self.register_buffer('y', torch.zeros(self.total_cells, requires_grad=False))
        # self.register_buffer('y_lambda', torch.zeros(m, requires_grad=False))
        # self.register_buffer('x_b', torch.zeros(self.total_cells, requires_grad=False))

        # self.register_buffer('phi', torch.zeros((self.m, self.n), requires_grad=False))
        # self.register_buffer('psi', torch.zeros((self.m, self.n), requires_grad=False))

    def _debug_log(self, tensor_dict):
        if self.debug:
            for name, t in tensor_dict.items():
                if type(t) in [int, float, bool]:
                    size = '-'
                else:
                    size = t.size()
                print([name, t, size])

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
        sigma = self.linear_a(x).view(self.m, 1).repeat(1, self.n)  # z_a (repeated for each cell, mxn)
        self._debug_log({'z_a': sigma})
        sigma += self.linear_b(x_b).view(self.m, self.n)  # z_b (sigma now dim mxn)
        self._debug_log({'z_a + z_b = sigma': sigma})
        return sigma

    def _inhibited_masking_and_prediction(self, sigma, phi):
        """
        Compute y_lambda
        """
        # Apply inhibition and shift to be non-neg
        pi = (1 - phi) * (sigma - sigma.min() + 1)
        self._debug_log({'pi': pi})

        # Group-wise max pooling
        lambda_i = self._group_max(pi)
        self._debug_log({'lambda_i': lambda_i})

        M_pi = topk_mask(pi, 1, dim=1)  # Mask: most active cell in group (total_cells)
        M_lambda = topk_mask(lambda_i, self.k, dim=0).view(self.m, 1).repeat(1, self.n)  # Mask: most active group (m)
        self._debug_log({'M_pi': M_pi, 'M_lambda': M_lambda})

        # Mask-based sparsening
        y = torch.tanh(M_pi * M_lambda * sigma)  # 1 x total_cells

        # Decode prediction through group-wise max bottleneck
        x_a_pred = self.linear_d(self._group_max(y))

        del M_pi
        del M_lambda

        return (y, x_a_pred)

    def _update_memory_and_inhibition(self, y, phi, psi):
        # Get updated psi (memory state), decay inactive inactive
        psi *= self.eps
        psi = torch.max(psi, y)

        # Update phi for next step (decay inactive cells)
        phi *= self.gamma
        phi = torch.max(phi, y)

        return (phi, psi)

    def forward(self, batch_x, x_b=None, phi=None, psi=None):
        """
        :param x: Input batch (batch_size, D_in)
        """
        if not x_b:
            x_b = batch_x.new_zeros(self.total_cells, requires_grad=False)
        if not phi:
            phi = batch_x.new_zeros((self.m, self.n), requires_grad=False)
        if not psi:
            psi = batch_x.new_zeros((self.m, self.n), requires_grad=False)

        # Can we vectorize across multiple batches?
        batch_size = batch_x.size(0)
        x_a_preds = []
        batch_x.new_zeros((batch_size, self.D_in))  # Conserve device

        for i, x in enumerate(batch_x):
            self._debug_log({'i': i, 'x': x})

            sigma = self._fc_weighted_ave(x.flatten(), x_b)
            self._debug_log({'sigma': sigma})

            y, x_a_pred = self._inhibited_masking_and_prediction(sigma, phi)
            self._debug_log({'y': y, 'x_a_pred': x_a_pred})

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
            x_b = x_b.detach()
            y = y.detach()
            sigma = sigma.detach()
            phi = phi.detach()
            psi = psi.detach()

            x_a_preds.append(x_a_pred)

        return (torch.stack(x_a_preds), x_b, phi, psi)


if __name__ == "__main__":
    batch_size, D_in = 50, 64

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(batch_size, D_in)
    y = torch.randn(batch_size, D_in)

    model = RSMLayer(D_in)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    for t in range(500):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
