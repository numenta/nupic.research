#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see htt"://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#

import string

import numpy as np
import torch

__all__ = [
    "CentroidFigure1B",
]


class CentroidFigure1B:
    """
    Mixin for generating data used to plot Figure 1B in the following CNS submission:

    "Going Beyond the Point Neuron: Active Dendrites and Sparse Representations for
    Continual Learning"; Karan Grewal, Jeremy Forest, Subutai Ahmad.
    """

    def setup_experiment(self, config):
        super().setup_experiment(config)

        if not hasattr(self, "ha_hook"):
            raise Exception("This experiment must use a Hidden Activations hook.")

    def validate(self, **kwargs):
        ret = super().validate(**kwargs)

        if 1 + self.current_task == self.num_tasks:

            # Validation loop for plotting purposes, where labels are unshared
            self.ha_hook.start_tracking()
            self.task_as_targets_eval()
            self.ha_hook.stop_tracking()

            # Save values for plotting

            # A key is just a unique identifier for this run, so that previously-saved
            # files aren't overwritten
            key = "".join(np.random.choice([s for s in string.ascii_lowercase], 4))

            for name, _, hidden_activations in self.ha_hook.get_statistics():
                targets = self.ha_targets[:self.ha_max_samples]

                filename = f"{self.num_tasks}_{name}_{key}.pt"
                torch.save(hidden_activations, f"x_{filename}")
                torch.save(targets, f"y_{filename}")

        return ret

    def task_as_targets_eval(self):
        """
        This method runs the evaluation loop similar to `self.evaluate_model`, but
        modified so that

        (1) target labels are converted to task labels, and
        (2) nothing is returned.
        """
        self.model.eval()
        with torch.no_grad():

            for data, target in self.val_loader:
                if isinstance(data, list):
                    data, _ = data
                data = data.flatten(start_dim=1)

                # This next line converts target labels to task labels, so test
                # examples can be grouped as desired
                target = target // self.num_classes_per_task

                data, target = data.to(self.device), target.to(self.device)

                # Select the context by comparing distances to all context prototypes
                context = torch.cdist(self.contexts, data)
                context = context.argmin(dim=0)
                context = self.contexts[context]

                # Perform forward pass to register hidden activations via hooks
                _ = self.model(data, context)

                if self.ha_hook.tracking:

                    # Targets were initialized on the cpu which could differ from the
                    # targets collected during the forward pass.
                    self.ha_targets = self.ha_targets.to(target.device)

                    # Concatenate and discard the older targets.
                    self.ha_targets = torch.cat([target, self.ha_targets], dim=0)
                    self.ha_targets = self.ha_targets[:self.ha_max_samples]

                else:
                    print("Warning: Hook tracking is off.")
