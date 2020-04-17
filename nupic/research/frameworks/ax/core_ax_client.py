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

import numbers

import numpy as np
from ax import Arm, Data
from ax.core import ObservationFeatures
from ax.storage.json_store.decoder import (
    generation_strategy_from_json,
    object_from_json,
)
from ax.storage.json_store.encoder import object_to_json


class CoreAxClient:
    """
    Similar to the AxClient, but relies on the caller to use more of the Ax Core
    API rather than wrapping it. This enables flexibility, for example it
    enables MultiObjective experiments.

    The AxClient exists because the Ax core API isn't really designed around
    using Ax as a service or as a data structure. The core API expectes you to
    create a custom Runner class, plug it into the Experiment, then call
    trial.run(). The AxClient and this class skip all of that and mimics the
    effect of trial.run() being called.
    """
    def __init__(self, experiment, generation_strategy, verbose=True):
        self.experiment = experiment
        self.generation_strategy = generation_strategy
        self.verbose = verbose

    def get_next_trial(self, model_gen_options=None):
        """
        @param model_gen_options (dict or None)
        Example value: {
             "acquisition_function_kwargs": {
                 "random_scalarization": True,
             },
        }
        """
        trial = self.experiment.new_trial(
            generator_run=self.generation_strategy.gen(
                experiment=self.experiment,
                n=1,
                pending_observations={
                    metric_name: [
                        ObservationFeatures(parameters=trial.arm.parameters,
                                            trial_index=np.int64(trial_index))
                        for trial_index, trial in self.experiment.trials.items()
                        if not trial.status.is_completed
                    ]
                    for metric_name in self.experiment.metrics
                },
                model_gen_options=model_gen_options,
            )
        )

        if self.verbose:
            print(f"Generated Ax trial {trial.index} with model "
                  f"{type(self.generation_strategy.model.model).__name__}")
            print(f"Marking Ax trial {trial.index} as running")
        trial.mark_running(no_runner_required=True)
        return trial.arm.parameters, trial.index

    def complete_trial(self, trial_index, raw_data, metadata=None):
        """
        This has more strict requirements of the raw_data than the AxClient, which
        simplifies this code.

        @param trial_index (int)
        The index returned by get_next_trial.

        @param raw_data (dict)
        Format: {"metric1": (mean1, sem1),
                 "metric2": (mean2, sem2)}
        If the sem is None, Ax will try to infer it.
        """
        if not isinstance(raw_data, dict) or any(isinstance(v, numbers.Number)
                                                 for v in raw_data.values()):
            # A more strict requirement than the AxClient (intentionally)
            raise ValueError(
                "CoreAxClient requires explicit metric names, means, and SEMs."
                f" You provided: {raw_data}")

        trial = self.experiment.trials.get(trial_index)
        trial._run_metadata = metadata if metadata is not None else {}
        self.experiment.attach_data(
            data=Data.from_evaluations(
                evaluations={trial.arm.name: raw_data},
                trial_index=trial.index,
            )
        )
        if self.verbose:
            print(f"Marking Ax trial {trial.index} as completed")
        trial.mark_completed()

    def attach_trial(self, parameters):
        self.experiment.search_space.check_membership(
            parameterization=parameters, raise_error=True
        )
        trial = self.experiment.new_trial().add_arm(Arm(parameters=parameters))
        trial.mark_running(no_runner_required=True)
        return trial.arm.parameters, trial.index

    def to_json_snapshot(self):
        """Serialize this `AxClient` to JSON to be able to interrupt and restart
        optimization and save it to file by the provided path.

        Returns:
            A JSON-safe dict representation of this `AxClient`.
        """
        return {
            "_type": self.__class__.__name__,
            "experiment": object_to_json(self.experiment),
            "generation_strategy": object_to_json(self.generation_strategy),
        }

    @staticmethod
    def from_json_snapshot(serialized):
        """Recreate a `CoreAxClient` from a JSON snapshot."""
        ax_client = CoreAxClient(
            experiment=object_from_json(serialized.pop("experiment")),
            generation_strategy=generation_strategy_from_json(
                generation_strategy_json=serialized.pop("generation_strategy")
            )
        )
        return ax_client

    def initialize_from_json_snapshot(self, serialized):
        other = CoreAxClient(
            experiment=object_from_json(serialized["experiment"]),
            generation_strategy=generation_strategy_from_json(
                generation_strategy_json=serialized["generation_strategy"]
            )
        )
        self.__dict__.update(other.__dict__)
