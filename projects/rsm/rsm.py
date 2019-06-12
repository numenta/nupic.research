import torch
from torch import nn


def topk(a, b):
    """
    Return a 1 for the top b elements in the last dim of a, 0 otherwise
    """
    values, indices = torch.topk(a, b)
    length = a.size()[-1]
    arr = a.new_zeros(length)  # Zeros, conserve device
    arr[indices[-1]] = 1
    return arr


class RSMLayer(torch.nn.Module):
    def __init__(self, D_in=28 * 28, m=200, n=6, k=25, gamma=0.5, eps=0.5):
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

        total_cells = m * n
        self.linear_a = nn.Linear(D_in, m)  # Input weights (shared per group / proximal)
        self.linear_b = nn.Linear(total_cells, total_cells)  # Recurrent weights (per cell)
        self.linear_d = nn.Linear(m, D_in)  # Decoding through bottleneck

        self.register_buffer('z_a', torch.zeros(m, requires_grad=False))
        self.register_buffer('z_b', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('pi', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('lambda_i', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('M_pi', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('M_lambda', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('sigma', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('y', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('x_b', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('phi', torch.zeros(total_cells, requires_grad=False))
        self.register_buffer('psi', torch.zeros(total_cells, requires_grad=False))

    def _group_max(self, activity):
        """
        :param activity: 1D vector of activity (m x n)

        Returns max cell activity in each group
        """
        max_res = activity.reshape(self.m, self.n).max(dim=1)
        return max_res.values

    def forward(self, batch_x):
        """
        :param x: Input batch (batch_size, D_in)
        """

        # Can we vectorize across multiple batches?
        batch_size = batch_x.size(0)
        x_a_pred = batch_x.new_zeros((batch_size, self.D_in))  # Conserve device

        for i, x in enumerate(batch_x):
            # Forward pass over one input

            self.z_a = self.linear_a(x.flatten())  # m
            self.z_b = self.linear_b(self.x_b)  # total_cells

            # z_a repeated (column weights shared by each cell in group)
            self.sigma = self.z_a.repeat(1, self.n) + self.z_b  # Weighted sum for each cell j in group i (mxn)

            # Apply inhibition and shift to be non-neg
            self.pi = (1 - self.phi) * (self.sigma - self.sigma.min() + 1)

            # Group-wise max pooling
            self.lambda_i = self._group_max(self.pi)

            self.M_pi = topk(self.pi, 1)  # Mask: most active cell in group (total_cells)
            self.M_lambda = topk(self.lambda_i, self.k)  # Mask: most active group (m)

            # Mask-based sparsening
            self.y = nn.functional.tanh(self.M_pi * self.M_lambda.repeat(self.n) * self.sigma)  # 1 x total_cells

            # Get updated psi (memory state), decay if inactive
            self.psi = torch.max(self.psi * self.eps, self.y)

            # Update phi for next step (decay if inactive)
            self.phi = torch.max(self.phi * self.gamma, self.y)

            # Update recurrent input / output x_b
            alpha = 1  # Normalizing scalar (force sum(x_b) == 1)
            self.x_b = alpha * self.psi

            # Detach hidden recurrent hidden layer to avoid 
            # "Trying to backward through the graph a second time" recursion error
            self.x_b = self.x_b.detach()

            # Decode prediction
            y_lambda = self._group_max(self.y)

            x_a_pred[i, :] = self.linear_d(y_lambda)
        return x_a_pred


if __name__ == "__main__":
    # batch_size is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    batch_size, D_in = 50, 64

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(batch_size, D_in)
    y = torch.randn(batch_size, D_in)

    # Construct our model by instantiating the class defined above
    model = RSMLayer(D_in)

    # Construct our loss function and an Optimizer. The call to model.parameters()
    # in the SGD constructor will contain the learnable parameters of the two
    # nn.Linear modules which are members of the model.
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
        loss.backward(retain_graph=True)
        optimizer.step()
