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


class GSCNoiseTest(object):
    """
    This mixin adds noise validation to standard GSC experiments.
    The noise validation will run at the end of the normal validation and
    add noise results to the validation results in the following format::

        results = {
            "total_correct": 0,
            "total_tested": 0,
            "mean_loss": 0,
            "mean_accuracy": 0,
            "noise": {
                "05": {
                    "total_correct": 0,
                    "total_tested": 0,
                    "mean_loss": 0,
                    "mean_accuracy": 0,
                },
                "10": { ... },
                ...
                "50: { ...}
            }
        }
    """
    def setup_experiment(self, config):
        """
        Configure noise test
        :param config:
            - noise_levels: list of noise levels to validate,
                            default ["05", "10", "15", "20", "25", "30", "45", "50"]
        """
        self.noise_levels = config.get("noise_levels", ["05", "10", "15", "20",
                                                        "25", "30", "45", "50"])
        super().setup_experiment(config)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].insert(0, "GSCNoiseTest")
        eo["create_loaders"].append("GSCNoiseTest")
        eo["validate"].append("GSCNoiseTest")

    def validate(self, loader=None):
        """
        Validate noise
        """
        results = super().validate(loader=loader)
        noise_results = {}
        self.logger.debug(f"Evaluating noise levels {self.noise_levels}")
        for noise in self.noise_levels:
            self.noise_loader.dataset.load_qualifier(noise)
            noise_results[noise] = super().validate(loader=self.noise_loader)
        results["noise"] = noise_results
        return results

    def create_loaders(self, config):
        """
        Create extra noise data loader
        """
        super().create_loaders(config)

        dataset_args = dict(config.get("dataset_args", {}))
        dataset_args.update(train=False, qualifiers=self.noise_levels)
        self.noise_loader = self.create_validation_dataloader(
            config={**config, "dataset_args": dataset_args})
