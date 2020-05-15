# ----------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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
#

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
# from nupic.research.frameworks.continuous_learning.dendrite_layers import DendriteLayer


def clear_labels(labels):
    indices = np.arange(11)
    out = np.delete(indices, labels)
    return out


def get_act(experiment):
    """ Gets network activations when presented with inputs for each class
    """

    layer_names = [p[0] for p in experiment.model.named_children()]

    act = {}

    def get_layer_act(name):
        def hook(model, input_, output):
            act[name] = output.detach().cpu().numpy()
        return hook

    cnt = 0
    for module in experiment.model:
        module.register_forward_hook(get_layer_act(layer_names[cnt]))
        cnt += 1

    outputs = []
    for k in range(1, 11):
        loader = experiment.test_loader[k]
        x, _ = next(iter(loader))
        experiment.model(x.cuda())
        outputs.append(act)
        act = {}

    return outputs


def dc_grad(model, kwinner_modules, duty_cycles, pct=90):
    all_modules = list(model.named_children())
    # module_dict = {k[0]: k[1] for k in all_modules}

    for module_name in kwinner_modules:
        if "kwinner" not in module_name:
            raise RuntimeError("Not a k-winner module")
        else:
            # module = module_dict[module_name]
            dc = torch.squeeze(duty_cycles[module_name])

            k = int((1 - pct / 100) * len(dc))
            _, inds = torch.topk(dc, k)

        module_num = module_name.split("_")[0][-1]
        module_type = module_name.split("_")[0][:-1]

        # find the module corresponding to the kwinners
        if module_type == "cnn":
            module_index = int(np.where(["cnn{}_cnn".format(module_num) in k[0]
                                         for k in all_modules])[0])
        elif module_type == "linear":
            module_index = int(np.where(["linear{}".format(module_num) in k[0]
                                         for k in all_modules])[0][0])

        weight_grads, bias_grads = [k.grad
                                    for k in all_modules[module_index][1].parameters()]

        with torch.no_grad():
            if module_type == "cnn":
                [weight_grads[ind, :, :, :].data.fill_(0.0) for ind in inds]
            elif module_type == "linear":
                [weight_grads[ind, :].fill_(0.0) for ind in inds]

            [bias_grads[ind].data.fill_(0.0) for ind in inds]


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# class ADA_fun(nn.Module):
#     def __init__(self, a=1, c=1, l=0.005):
#         super(ADA_fun, self).__init__()
#         self.a = a
#         self.c = c
#         self.l = l

#     def forward(self, x):
#         neg_relu = torch.clamp(x, max=0)
#         ADA = F.relu(x) * torch.exp(-x * self.a + self.c)
#         ADA_l = self.l * neg_relu + ADA
#         return ADA_l

