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
import copy
import socket
from contextlib import closing

from torch.optim.lr_scheduler import OneCycleLR

from nupic.research.frameworks.pytorch.lr_scheduler import ComposedLRScheduler


def create_lr_scheduler(optimizer, lr_scheduler_class, lr_scheduler_args,
                        steps_per_epoch):
    """
    Configure learning rate scheduler

    :param optimizer:
        Wrapped optimizer
    :param lr_scheduler_class:
        LR scheduler class to use. Must inherit from _LRScheduler
    :param lr_scheduler_args:
        LR scheduler class constructor arguments
    :param steps_per_epoch:
        The total number of batches in the epoch.
        Only used if lr_scheduler_class is :class:`ComposedLRScheduler` or
        :class:`OneCycleLR`
    """
    if issubclass(lr_scheduler_class, OneCycleLR):
        # Update OneCycleLR parameters
        lr_scheduler_args = copy.deepcopy(lr_scheduler_args)
        lr_scheduler_args.update(steps_per_epoch=steps_per_epoch)
    elif issubclass(lr_scheduler_class, ComposedLRScheduler):
        # Update ComposedLRScheduler parameters
        lr_scheduler_args = copy.deepcopy(lr_scheduler_args)
        schedulers = lr_scheduler_args.get("schedulers", None)
        if schedulers is not None:
            # Convert dict from ray/json {str:dict} style to {int:dict}
            schedulers = {int(k): v for k, v in schedulers.items()}

            # Update OneCycleLR "steps_per_epoch" parameter
            for _, item in schedulers.items():
                lr_class = item.get("lr_scheduler_class", None)
                if lr_class is not None and issubclass(lr_class, OneCycleLR):
                    lr_args = item.get("lr_scheduler_args", {})
                    lr_args.update(steps_per_epoch=steps_per_epoch)
            lr_scheduler_args["schedulers"] = schedulers
        lr_scheduler_args["steps_per_epoch"] = steps_per_epoch

    return lr_scheduler_class(optimizer, **lr_scheduler_args)


def get_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        # bind on port 0 - kernel will select an unused port
        s.bind(("", 0))
        # removed socket.SO_REUSEADDR arg
        # TCP error due to two process with same rank in same port - maybe a fix
        return s.getsockname()[1]


def get_node_ip_address():
    """
    Determine the IP address of the local node.
    """
    return socket.gethostbyname(socket.getfqdn())
