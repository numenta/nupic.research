#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
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

from pprint import pformat, pprint

from ray.tune import Trainable

from nupic.research.frameworks.sigopt import SigOptExperiment
from nupic.research.frameworks.trainables import (
    DistributedTrainable,
    RemoteProcessTrainable,
)


class SigOptTrainableMixin:
    """
    This class updates the config using SigOpt before the models and workers are
    instantiated, and updates the result using SigOpt once training completes.
    """

    def _process_config(self, config):
        """
        :param config:
            Dictionary configuration of the trainable

            - sigopt_experiment_id: id of experiment
            - sigopt_config: dict to specify configuration of sigopt experiment
            - sigopt_experiment_class: class inherited from `SigoptExperiment` which
                                       characterizes how the trainable will get and
                                       utilize suggestions

        """
        # Update the config through SigOpt.
        self.sigopt = None
        if "sigopt_config" in config:
            assert config.get("sigopt_experiment_id", None) is not None

            # Check for user specified sigopt-experiment class.
            experiment_class = config.get(
                "sigopt_experiment_class", SigOptExperiment)
            assert issubclass(experiment_class, SigOptExperiment)

            # Instantiate experiment.
            self.sigopt = experiment_class(
                experiment_id=config["sigopt_experiment_id"],
                sigopt_config=config["sigopt_config"])

            self.logger.info(
                f"Sigopt execution order: {pformat(self.sigopt.get_execution_order())}")

            # Get suggestion and update config.
            self.suggestion = self.sigopt.get_next_suggestion()
            self.sigopt.update_config_with_suggestion(config, self.suggestion)
            print("SigOpt suggestion: ", self.suggestion)
            print("Config after Sigopt:")
            pprint(config)
            self.epochs = config["epochs"]

            # Get names of performance metrics.
            assert "metrics" in config["sigopt_config"]
            self.metric_names = [
                metric["name"] for metric in config["sigopt_config"]["metrics"]
            ]
            assert len(self.metric_names) > 0, \
                "For now, we only update the observation if a metric is present."

    def _process_result(self, result):
        """
        Update sigopt with the new result once we're at the end of training.
        """

        super()._process_result(result)

        if self.sigopt is not None:
            result["early_stop"] = result.get("early_stop", 0.0)
            if self.iteration >= self.epochs - 1:
                result["early_stop"] = 1.0
                # check that all metrics are present
                print(result)
                for name in self.metric_names:
                    if result[name] is not None:
                        self.logger.info(f"Updating observation {name} with value=",
                                         result[name])
                    else:
                        self.logger.warning(f"No value: {name}")

                # Collect and report relevant metrics.
                values = [
                    dict(name=name, value=result[name])
                    for name in self.metric_names
                ]
                self.sigopt.update_observation(self.suggestion, values=values)
                print("Full results: ")
                pprint(result)


class SigOptRemoteProcessTrainable(SigOptTrainableMixin, RemoteProcessTrainable):
    pass


class SigOptDistributedTrainable(SigOptTrainableMixin, DistributedTrainable):
    pass

