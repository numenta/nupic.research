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


import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn


def clear_labels(labels, length=11):
    indices = np.arange(length)
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
            module_index = int(
                np.where(["cnn{}_cnn".format(module_num) in k[0] for k in all_modules])[
                    0
                ]
            )
        elif module_type == "linear":
            module_index = int(
                np.where(["linear{}".format(module_num) in k[0] for k in all_modules])[
                    0
                ][0]
            )

        weight_grads, bias_grads = [
            k.grad for k in all_modules[module_index][1].parameters()
        ]

        with torch.no_grad():
            if module_type == "cnn":
                [weight_grads[ind, :, :, :].data.fill_(0.0) for ind in inds]
            elif module_type == "linear":
                [weight_grads[ind, :].fill_(0.0) for ind in inds]

            [bias_grads[ind].data.fill_(0.0) for ind in inds]


def train_model(
    model,
    loader,
    optimizer,
    device,
    sample_fraction=None,
    normalize_input=False,
    freeze_modules=None,
    module_inds=None,
    freeze_output=False,
    layer_type="dense",
    linear_number=2,
    output_indices=None,
    criterion=F.nll_loss,
    batches_in_epoch=sys.maxsize,
    pre_batch_callback=None,
    post_batch_callback=None,
    progress_bar=None,
    combine_data=False,
):

    model.train()
    # Use asynchronous GPU copies when the memory is pinned
    # See https://pytorch.org/docs/master/notes/cuda.html
    async_gpu = loader.pin_memory
    if progress_bar is not None:
        loader = tqdm(loader, **progress_bar)
        # update progress bar total based on batches_in_epoch
        if batches_in_epoch < len(loader):
            loader.total = batches_in_epoch

    t0 = time.time()

    if sample_fraction is not None:
        num_batches = int(loader.dataset.targets.shape[0] / loader.batch_size)
        max_batch_idx = int(sample_fraction * num_batches)
    else:
        max_batch_idx = batches_in_epoch

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= max_batch_idx:
            break

        if normalize_input:
            data = norm_input(data)

        num_images = len(target)
        data = data.to(device, non_blocking=async_gpu)
        target = target.to(device, non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        optimizer.zero_grad()
        if combine_data:
            output = model(data, target)
        else:
            output = model(data)

        loss = criterion(output, target)
        del data, target, output

        t2 = time.time()

        loss.backward()

        if freeze_modules is not None:
            cnt = 0
            # with torch.no_grad():
            for module in freeze_modules:
                freeze_grads(module, module_inds[cnt])
                cnt += 1

        if freeze_output:
            freeze_output_layer(
                model,
                output_indices,
                layer_type=layer_type,
                linear_number=linear_number,
            )

        t3 = time.time()
        optimizer.step()  # step
        t4 = time.time()

        if post_batch_callback is not None:
            time_string = (
                "Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                + "weight update: {:.3f}s"
            ).format(t1 - t0, t2 - t1, t3 - t2, t4 - t3)
            post_batch_callback(
                model=model,
                loss=loss.detach(),
                batch_idx=batch_idx,
                num_images=num_images,
                time_string=time_string,
            )
        del loss
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()


def freeze_output_layer(model, indices, layer_type="dense", linear_number=2):
    """ Freeze output layer gradients of specific classes for classification.
    :param layer_type: can be "dense" (i.e. model.output) or "kwinner"
    :param linear_number: "linear" module number for k winner
    (e.g. linear1_kwinners, linear2_kwinners etc.)
    """
    if layer_type == "dense":
        with torch.no_grad():
            [model.output.weight.grad.data[index, :].fill_(0.0) for index in indices]
            [model.output.bias.grad.data[index].fill_(0.0) for index in indices]

    elif layer_type == "kwinner":
        module_dict = {k[0]: k[1] for k in model.named_parameters()}
        with torch.no_grad():
            [
                module_dict["linear{}.module.weight".format(linear_number)]
                .grad.data[index, :]
                .fill_(0.0)
                for index in indices
            ]
            [
                module_dict["linear{}.module.bias".format(linear_number)]
                .grad.data[index]
                .fill_(0.0)
                for index in indices
            ]

    else:
        raise AssertionError("layer_type must be ''dense'' or ''kwinner''")


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def freeze_grads(module, inds):
    with torch.no_grad():
        weight_grads, bias_grads = list(module.parameters())
        if len(weight_grads.shape) > 2:
            [weight_grads.data[index, :, :, :].fill_(0.0) for index in inds]
        else:
            [weight_grads.data[index, :].fill_(0.0) for index in inds]
        [bias_grads.data[index].fill_(0.0) for index in inds]


def split_inds(module, no_splits, no_units=None):
    if no_units is None:
        weight, bias = list(module.parameters())
        no_units = len(bias)
    inds = torch.randperm(no_units)
    split_len = int(no_units / no_splits)
    inds = torch.split(inds, split_len)

    return [k.numpy() for k in inds]


def norm_input(x):
    x_ = torch.zeros_like(x)
    for i in range(x.shape[0]):
        enum = x[i, ...] - torch.mean(x[i, ...])
        # out = enum / torch.std(x[i, ...])
        x_[i, ...] = enum

    return x_
