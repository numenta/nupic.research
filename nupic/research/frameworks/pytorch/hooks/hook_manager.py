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

from .base import TrackStatsHookBase


class ModelHookManager:
    """
    This class registers and manages a set of hooks of subclassed from
    `TrackStatsHookBase`. The given hook is registered on all modules within
    'named_modules'.

    Tracking is started and stopped for all hooks via `self.start_tracking()` and
    `self.stop_tracking()`. Alternatively, this class can be used a context manager to
    automate these calls. For example,

    ```
    with hook_manager as hooks:
        ... # Train here
         stats = hooks.get_statitics()
    ```

    .. _filter_modules: nupic.research.frameworks.pytorch.model_utils.filter_modules

    :param named_modules: dict mapping names to modules
    :param hook_class: class subclassed from `TrackStatsHookBase`
    :param hook_type: whether to register the hook as "forward" or "backward"
                      or "pre_forward"
    """

    def __init__(self, named_modules, hook_class, hook_type="forward"):

        assert hook_type in ["forward", "backward", "pre_forward"]
        assert issubclass(hook_class, TrackStatsHookBase)

        # Register the hooks via class method.
        tracked_vals = self.register_storage_hooks(named_modules,
                                                   hook_class=hook_class,
                                                   hook_type=hook_type)

        # These are the functions that called every forward or backward pass.
        self.hooks = tracked_vals[0]

        # These are handles to the hooks; PyTorch lets the user unregister
        # hooks through these handles.
        self._hook_handles = tracked_vals[1]

        # These are the filtered modules that will be tracked.
        self.tracked_modules = tracked_vals[2]

    def __enter__(self):
        """Start tracking when `with` is called."""
        self.start_tracking()
        return self

    def __exit__(self, *args):
        """Stop tracking when `with` block is left."""
        self.stop_tracking()

    @classmethod
    def register_storage_hooks(cls, named_modules, hook_class, hook_type="forward"):
        """
        Register hook on each module in 'named_modules'.

        :param named_modules: dict mapping names to modules
        :param hook_class: class subclassed from `TrackStatsHookBase`
        :param hook_type: whether to register the hook as "forward" or "backward"
                          or "pre_forward"
        """
        assert hook_type in ["forward", "backward", "pre_forward"]

        hooks = []
        handles = []
        tracked_modules = dict()

        # Register hooks on the modules.
        for n, m in named_modules.items():

            hook = hook_class(name=n)
            if hook_type == "forward":
                handle = m.register_forward_hook(hook)
            elif hook_type == "pre_forward":
                handle = m.register_forward_pre_hook(hook)
            else:
                handle = m.register_backward_hook(hook)

            hooks.append(hook)
            handles.append(handle)
            tracked_modules[n] = m

        return hooks, handles, tracked_modules

    def start_tracking(self):
        for hook in self.hooks:
            hook.start_tracking()

    def stop_tracking(self):
        for hook in self.hooks:
            hook.stop_tracking()

    def get_statistics(self):
        """
        This returns a generator with elements
        `(name, module, statistic_0, ..., statistic_n)`.
        """
        return (
            (name, module, *hook.get_statistics())
            for (name, module), hook in zip(self.tracked_modules.items(), self.hooks)
        )

    def remove_hooks(self):
        """
        Remove all hooks from the model and stop tracking statistics.
        """
        for handle in self._hook_handles:
            handle.remove()

        self.hooks = []
        self._hook_handles = []
        self.tracked_modules = dict()
