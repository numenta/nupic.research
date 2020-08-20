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


class ReduceLRAfterTask:
    """
    Freeze after params after task k.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.task_when_reduce_lr = config.get("task_when_reduce_lr", 0)
        self.new_lr = config.get("new_lr", 1e-3)

    def run_task(self):
        """Run outer loop over tasks"""
        # configure the sampler to load only samples from current task
        if self.current_task >= self.task_when_reduce_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.new_lr

        super().run_task()

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("Freeze after task params")
        eo["run_task"].insert(0, "Reduce LR for all weights")
        return eo
