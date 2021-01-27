# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
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

import abc


class TrackStatsHookBase(metaclass=abc.ABCMeta):
    """
    This is a base class for tracking `hooks`_ via forward or backward passes. This
    provides a simple API to turn off and on tracking (say to only track during
    certain parts of training or just during validation) and then later retrieve the
    recorded statistics. Tracking is started and ended as needed via
    `self.start_tracking()` and `self.end_tracking()`. And recorder statistics are
    retrieved via `self.get_statistics()`.

    ```
    # Init your model and register the hook.
    module = Model(...)
    hook = TrackStatsHook()
    module.register_forward_hook(hook)

    # Train here.
    ...

    # Validate and track stats here.
    hook.start_tracking()
    for x, y in val_loader:
        ...
    hook.stop_tracking()
    validation_stats = hook.get_statistics()
    ```

    Here is a tutorial on forward and backward hooks:
    .. _hooks: https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html

    Read here for more info on each type of module hook:
    - register_backward_hook:
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_backward_hook
    - register_forward_hook
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
    - register_forward_pre_hook
        https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook

    Read here for more info on a tensor hook:
    - register_hook
        https://pytorch.org/docs/stable/autograd.html#torch.Tensor.register_hook
    """

    def __init__(self, name=None):
        self.name = name
        self._tracking = False

    def start_tracking(self):
        self._tracking = True

    def stop_tracking(self):
        self._tracking = False

    @abc.abstractmethod
    def get_statistics(self):
        """Returns tuple of recorded statistics."""
        raise NotImplementedError

    @abc.abstractmethod
    def __call__(self, *args):
        """
        This method will be called on either the forward or backward pass depending on
        how the hook is registered. Accordingly, the arguments will vary. A forward pass
        will have a signature of `__call__(module, forward_input, forward_ouput)` and,
        similarly, a backward pass will have a signature of `__call__(module,
        backward_input, backward_output)`. As well, a forward pre-hook will have a
        signature of `__call__(module, forward_input)` and a gradient hook on a tensor
        will have `__call__(grad)`.
        """
        raise NotImplementedError
