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

from pprint import pformat, pprint

from nupic.research.frameworks.ray.trainables import RemoteProcessTrainable
from nupic.research.frameworks.sigopt import SigOptExperiment


class MetaCLSigOptRemoteProcessTrainable(RemoteProcessTrainable):
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
                "sigopt_experiment_class", MetaCLSigOptExperiment)
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

    def _process_result(self, result):
        """
        This overwrites the _process_result of the parent mixin.

        Update sigopt with the new result once we're at the end of training.
        """

        super()._process_result(result)

        if self.sigopt is not None:
            # Default value for ealy stop: 0.0 (i.e. don't stop)
            result["early_stop"] = result.get("early_stop", 0.0)

            # Identify if the experiment should be stopped early.
            # If so, update with result.
            if self.iteration >= self.epochs - 1:
                result["early_stop"] = 1.0

                # Calculate mean meta-testing accuracy.
                mean_test_test_acc = 0
                total = 0
                for name, val in result.items():
                    if "mean_test_test" in name:
                        mean_test_test_acc += val
                        total += 1
                assert total > 0, "No meta-testing accuracies reported."
                mean_test_test_acc = mean_test_test_acc / total

                # Collect and report relevant metrics.
                values = [
                    dict(name="mean_test_test_acc", value=mean_test_test_acc)
                ]

                self.sigopt.update_observation(self.suggestion, values=values)
                print("Full results: ")
                pprint(result)


class MetaCLSigOptExperiment(SigOptExperiment):
    def update_config_with_suggestion(self, config, suggestion):
        """
        Given a SigOpt suggestion, update the learning rate for the inner and outer
        loops.

        :param config:
            - optimizer_args: dict of optimizer arguments
        :param suggestion:
            - assignments (all optional)
                - log10_inner_lr: learning rate for inner loop
                - log10_outer_lr: learning rate for outer loop
        """
        super().update_config_with_suggestion(config, suggestion)

        assignments = suggestion.assignments

        assert "optimizer_args" in config
        optimizer_args = config["optimizer_args"]

        if "log10_outer_lr" in assignments:
            optimizer_args["lr"] = 10 ** assignments["log10_outer_lr"]

        if "log10_inner_lr" in assignments:
            config["adaptation_lr"] = 10 ** assignments["log10_inner_lr"]

    @classmethod
    def get_execution_order(cls):
        return dict(
            update_config_with_suggestion=[
                "MetaCLSigOptExperiment.update_config_with_suggestion"
            ],
        )
