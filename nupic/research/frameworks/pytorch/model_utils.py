# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2018, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import gzip
import pickle
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_model(
    model,
    loader,
    optimizer,
    device,
    freeze_params=None,
    criterion=F.nll_loss,
    complexity_loss_fn=None,
    batches_in_epoch=sys.maxsize,
    pre_batch_callback=None,
    post_batch_callback=None,
    transform_to_device_fn=None,
    progress_bar=None,
):
    """Train the given model by iterating through mini batches. An epoch ends
    after one pass through the training set, or if the number of mini batches
    exceeds the parameter "batches_in_epoch".

    :param model: pytorch model to be trained
    :type model: torch.nn.Module
    :param loader: train dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param optimizer: Optimizer object used to train the model.
           This function will train the model on every batch using this optimizer
           and the :func:`torch.nn.functional.nll_loss` function
    :param batches_in_epoch: Max number of mini batches to train.
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device
    :param freeze_params: List of parameters to freeze at specified indices
     For each parameter in the list:
     - parameter[0] -> network module
     - parameter[1] -> weight indices
    :type param: list or tuple
    :param criterion: loss function to use
    :type criterion: function
    :param complexity_loss_fn: a regularization term for the loss function
    :type complexity_loss_fn: function
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters: model, batch_idx
    :type post_batch_callback: function
    :param pre_batch_callback: Callback function to be called before every batch
                               with the following parameters: model, batch_idx
    :type pre_batch_callback: function
    :param transform_to_device_fn: Function for sending data and labels to the
                                   device. This provides an extensibility point
                                   for performing any final transformations on
                                   the data or targets, and determining what
                                   actually needs to get sent to the device.
    :type transform_to_device_fn: function
    :param progress_bar: Optional :class:`tqdm` progress bar args.
                         None for no progress bar
    :type progress_bar: dict or None

    :return: mean loss for epoch
    :rtype: float
    """
    model.train()
    # Use asynchronous GPU copies when the memory is pinned
    # See https://pytorch.org/docs/master/notes/cuda.html
    async_gpu = loader.pin_memory
    if progress_bar is not None:
        loader = tqdm(loader, **progress_bar)
        # update progress bar total based on batches_in_epoch
        if batches_in_epoch < len(loader):
            loader.total = batches_in_epoch

    # Check if training with Apex Mixed Precision
    # FIXME: There should be another way to check if 'amp' is enabled
    use_amp = hasattr(optimizer, "_amp_stash")
    try:
        from apex import amp
    except ImportError:
        if use_amp:
            raise ImportError(
                "Mixed precision requires NVIDA APEX."
                "Please install apex from https://www.github.com/nvidia/apex")

    t0 = time.time()
    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= batches_in_epoch:
            break

        num_images = len(target)
        if transform_to_device_fn is None:
            data = data.to(device, non_blocking=async_gpu)
            target = target.to(device, non_blocking=async_gpu)
        else:
            data, target = transform_to_device_fn(data, target, device,
                                                  non_blocking=async_gpu)
        t1 = time.time()

        if pre_batch_callback is not None:
            pre_batch_callback(model=model, batch_idx=batch_idx)

        optimizer.zero_grad()
        output = model(data)
        error_loss = criterion(output, target)
        del data, target, output

        complexity_loss = (complexity_loss_fn(model)
                           if complexity_loss_fn is not None
                           else None)

        loss = (error_loss + complexity_loss
                if complexity_loss is not None
                else error_loss)
        # print(f"Debugging loss: {loss:.4f}")
        t2 = time.time()
        if use_amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        del loss

        if freeze_params is not None:
            with torch.no_grad():
                for param in freeze_params:
                    param_module = param[0]
                    param_indices = param[1]
                    param_module.grad[param_indices, :] = 0.0

        t3 = time.time()
        optimizer.step()
        t4 = time.time()

        if post_batch_callback is not None:
            time_string = ("Data: {:.3f}s, forward: {:.3f}s, backward: {:.3f}s,"
                           + "weight update: {:.3f}s").format(t1 - t0, t2 - t1, t3 - t2,
                                                              t4 - t3)
            post_batch_callback(model=model,
                                error_loss=error_loss.detach(),
                                complexity_loss=(complexity_loss.detach()
                                                 if complexity_loss is not None
                                                 else None),
                                batch_idx=batch_idx,
                                num_images=num_images,
                                time_string=time_string)
        del error_loss, complexity_loss
        t0 = time.time()

    if progress_bar is not None:
        loader.n = loader.total
        loader.close()


def evaluate_model(
    model,
    loader,
    device,
    batches_in_epoch=sys.maxsize,
    criterion=F.nll_loss,
    complexity_loss_fn=None,
    progress=None,
    post_batch_callback=None,
    transform_to_device_fn=None,
):
    """Evaluate pre-trained model using given test dataset loader.

    :param model: Pretrained pytorch model
    :type model: torch.nn.Module
    :param loader: test dataset loader
    :type loader: :class:`torch.utils.data.DataLoader`
    :param device: device to use ('cpu' or 'cuda')
    :type device: :class:`torch.device`
    :param batches_in_epoch: Max number of mini batches to test on.
    :type batches_in_epoch: int
    :param criterion: loss function to use
    :type criterion: function
    :param complexity_loss_fn: a regularization term for the loss function
    :type complexity_loss_fn: function
    :param progress: Optional :class:`tqdm` progress bar args. None for no progress bar
    :type progress: dict or None
    :param post_batch_callback: Callback function to be called after every batch
                                with the following parameters:
                                batch_idx, target, output, pred
    :type post_batch_callback: function
    :param transform_to_device_fn: Function for sending data and labels to the
                                   device. This provides an extensibility point
                                   for performing any final transformations on
                                   the data or targets, and determining what
                                   actually needs to get sent to the device.
    :type transform_to_device_fn: function

    :return: dictionary with computed "mean_accuracy", "mean_loss", "total_correct".
    :rtype: dict
    """
    model.eval()
    total = 0

    # Perform accumulation on device, avoid paying performance cost of .item()
    loss = torch.tensor(0., device=device)
    correct = torch.tensor(0, device=device)

    async_gpu = loader.pin_memory

    if progress is not None:
        loader = tqdm(loader, **progress)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx >= batches_in_epoch:
                break

            if transform_to_device_fn is None:
                data = data.to(device, non_blocking=async_gpu)
                target = target.to(device, non_blocking=async_gpu)
            else:
                data, target = transform_to_device_fn(data, target, device,
                                                      non_blocking=async_gpu)

            output = model(data)
            loss += criterion(output, target, reduction="sum")
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum()
            total += len(data)

            if post_batch_callback is not None:
                post_batch_callback(batch_idx=batch_idx, target=target, output=output,
                                    pred=pred)

        complexity_loss = (complexity_loss_fn(model)
                           if complexity_loss_fn is not None
                           else None)

    if progress is not None:
        loader.close()

    correct = correct.item()
    loss = loss.item()

    result = {
        "total_correct": correct,
        "total_tested": total,
        "mean_loss": loss / total if total > 0 else 0,
        "mean_accuracy": correct / total if total > 0 else 0,
    }

    if complexity_loss is not None:
        result["complexity_loss"] = complexity_loss.item()

    return result


def aggregate_eval_results(results):
    """Aggregate multiple results from evaluate_model into a single result.

    This function ignores fields that don't need aggregation. To get the
    complete result dict, start with a deepcopy of one of the result dicts,
    as follows:
        result = copy.deepcopy(results[0])
        result.update(aggregate_eval_results(results))

    :param results:
        A list of return values from evaluate_model.
    :type results: list

    :return:
        A single result dict with evaluation results aggregated.
    :rtype: dict
    """
    correct = sum(result["total_correct"] for result in results)
    total = sum(result["total_tested"] for result in results)
    if total == 0:
        loss = 0
        accuracy = 0
    else:
        loss = sum(result["mean_loss"] * result["total_tested"]
                   for result in results) / total
        accuracy = correct / total

    return {
        "total_correct": correct,
        "total_tested": total,
        "mean_loss": loss,
        "mean_accuracy": accuracy,
    }


def set_random_seed(seed, deterministic_mode=True):
    """
    Set pytorch, python random, and numpy random seeds (these are all the seeds we
    normally use).

    :param seed:  (int) seed value
    :param deterministic_mode: (bool) If True, then even on a GPU we'll get more
           deterministic results, though performance may be slower. See:
           https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available() and deterministic_mode:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_nonzero_params(model):
    """
    Count the total number of non-zero weights in the model, including bias weights.
    """
    total_nonzero_params = 0
    total_params = 0
    for param in model.parameters():
        total_nonzero_params += param.data.nonzero().size(0)
        total_params += param.data.numel()

    return total_params, total_nonzero_params


def serialize_state_dict(fileobj, state_dict, compresslevel=3):
    """
    Serialize the state dict to file object
    :param fileobj: file-like object such as :class:`io.BytesIO`
    :param state_dict: state dict to serialize. Usually the dict returned by
                       module.state_dict() but it can be any state dict.
    :param compresslevel: compression level for gzip (lower equals faster but
                          less compression).
   """
    with gzip.GzipFile(fileobj=fileobj, mode="wb", compresslevel=compresslevel) as fout:
        torch.save(state_dict, fout, pickle_protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_state_dict(fileobj, device=None):
    """
    Deserialize state dict saved via :func:`_serialize_state_dict` from
    the given file object
    :param fileobj: file-like object such as :class:`io.BytesIO`
    :param device: Device to map tensors to
    :return: the state dict stored in the file object
    """
    try:
        with gzip.GzipFile(fileobj=fileobj, mode="rb") as fin:
            state_dict = torch.load(fin, map_location=device)
    except OSError:
        # FIXME: Backward compatibility with old uncompressed checkpoints
        state_dict = torch.load(fileobj, map_location=device)
    return state_dict


def get_module_attr(module, name):
    """
    Get model attribute of torch.nn.Module given its name.
    Ex:
    ```
    name = "features.stem.weight"
    weight = get_module_attr(net, name)
    ```
    """

    # Split off the last sub-name.
    # Ex. "features.stem.weight" -> ("features.stem", "weight")
    parent_name, sub_name = (name.split(".", 1) + [None])[0:2]

    if sub_name is not None:
        parent_module = _get_sub_module(module, parent_name)
        sub_module = get_module_attr(parent_module, sub_name)
        return sub_module
    else:
        sub_module = _get_sub_module(module, parent_name)
        return sub_module


def set_module_attr(module, name, value):
    """
    Set model attribute of torch.nn.Module given its name.
    Ex:
    ```
    name = "features.stem.weight"
    weight = Parameter(...)
    set_module_attr(net, name, weight)
    ```
    """

    # Split name: pytorch convention uses "." for each child module.
    all_names = name.split(".")

    # Get all names except the last.
    # Ex. "features.stem.weight" -> "features.stem"
    parents_names = all_names[:-1]

    if not parents_names:
        setattr(module, name, value)

    else:
        # Get the parent module of the last child module.
        parent_name = ".".join(parents_names)
        parent_module = get_module_attr(module, parent_name)

        # Set the new value of the last child module.
        child_name = all_names[-1]
        setattr(parent_module, child_name, value)


def _get_sub_module(module, name):
    """
    Gets a submodule either by name or index - pytorch either uses names for module
    attributes (e.g. "module.classifier") or indices for sequential models
    (e.g. `module[0]`).
    ```
    """
    if name.isdigit():
        return module[int(name)]
    else:
        return getattr(module, name)
