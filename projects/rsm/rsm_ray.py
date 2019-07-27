#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2019, Numenta, Inc.  Unless you have an agreement
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

import argparse
import os

import ray
import torch
from ray import tune
from ray.tune.logger import CSVLogger, JsonLogger

from nupic.research.frameworks.pytorch.tf_tune_utils import TFLoggerPlus
from nupic.research.support.elastic_logger import ElasticsearchLogger
from nupic.research.support.parse_config import parse_config
from rsm_experiment import RSMExperiment


def trial_name_string(trial):
    """
    Args:
      trial (Trial): A generated trial object.

    Returns:
      trial_name (str): String representation of Trial.
    """
    s = str(trial)
    chars = "{}[]() ,="
    for c in chars:
        s = s.replace(c, "_")

    if len(s) > 85:
        s = s[0:75] + "_" + s[-10:]
    return s


class RSMTune(RSMExperiment, tune.Trainable):
    """ray.tune trainable class for running small RSM models:

    - Override _setup to reset the experiment for each trial.
    - Override _train to train and evaluate each epoch
    - Override _save and _restore to serialize the model
    """

    def __init__(self, config=None, logger_creator=None):
        RSMExperiment.__init__(self, config=config)
        tune.Trainable.__init__(self, config=config, logger_creator=logger_creator)

    def _setup(self, config):
        """Custom initialization.

        Args:
            config (dict): Hyperparameters and other configs given.
                Copy of `self.config`.
        """
        self.model_setup(config)

    def _train(self):
        """Implement train() for a single epoch.

        Returns:
            A dict that describes training progress.
        """
        ret = self.train_epoch(self._iteration)
        return ret

    def _save(self, checkpoint_dir):
        return self.model_save(checkpoint_dir)

    def _restore(self, checkpoint):
        """Subclasses should override this to implement restore().

        Args:
            checkpoint (str | dict): Value as returned by `_save`.
                If a string, then it is the checkpoint path.
        """
        self.model_restore(checkpoint)

    def _stop(self):
        """Subclasses should override this for any cleanup on stop."""
        if self._iteration < self.iterations:
            print("RSMTune: stopping early at epoch {}".format(self._iteration))
        self.model_cleanup()


@ray.remote
def run_experiment(config, trainable):
    """Run a single tune experiment in parallel as a "remote" function.

    :param config: The experiment configuration
    :type config: dict
    :param trainable: tune.Trainable class with your experiment
    :type trainable: :class:`ray.tune.Trainable`
    """
    # Stop criteria. Default to total number of iterations/epochs
    stop_criteria = {"training_iteration": config.get("iterations")}
    stop_criteria.update(config.get("stop", {}))
    no_gpu = config.get("num_gpus") == 0
    tune.run(
        trainable,
        name=config["name"],
        local_dir=config["path"],
        stop=stop_criteria,
        config=config,
        num_samples=config.get("repetitions", 1),
        trial_name_creator=tune.function(trial_name_string),
        trial_executor=config.get("trial_executor", None),
        checkpoint_at_end=config.get("checkpoint_at_end", False),
        checkpoint_freq=config.get("checkpoint_freq", 0),
        upload_dir=config.get("upload_dir", None),
        sync_function=config.get("sync_function", None),
        resume=config.get("resume", False),
        reuse_actors=config.get("reuse_actors", False),
        loggers=(JsonLogger, CSVLogger, TFLoggerPlus, ElasticsearchLogger),
        verbose=config.get("verbose", 0),
        resources_per_trial={
            # With lots of trials, optimal seems to be 0.5, or 2 trials per GPU
            # If num trials <= num GPUs, 1.0 is better
            "cpu": 1,
            "gpu": 0 if no_gpu else config.get("gpu_percentage", 0.5),
        },
    )


def parse_options():
    """parses the command line options for different settings."""
    optparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    optparser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=open,
        default="experiments.cfg",
        help="your experiments config file",
    )
    optparser.add_argument(
        "-n",
        "--num_cpus",
        dest="num_cpus",
        type=int,
        default=os.cpu_count() - 1,
        help="number of cpus you want to use",
    )
    optparser.add_argument(
        "-g",
        "--num_gpus",
        dest="num_gpus",
        type=int,
        default=torch.cuda.device_count(),
        help="number of gpus you want to use",
    )
    optparser.add_argument(
        "-e",
        "--experiment",
        action="append",
        dest="experiments",
        help="run only selected experiments, by default run all experiments in "
        "config file.",
    )
    optparser.add_argument(
        "-r", "--resume", dest="resume", help="resume from checkpoint if found"
    )
    optparser.add_argument(
        "-p", "--predict", dest="predict", help="run prediction on trained model"
    )
    optparser.add_argument(
        "-l",
        "--plot_gradients",
        dest="plot_gradients",
        help="Plot gradients for debugging",
        default=False,
    )
    optparser.add_argument(
        "-v", "--verbose", dest="verbose", help="Verbosity", default=0
    )

    return optparser.parse_args()


if __name__ == "__main__":
    # Load and parse command line option and experiment configurations
    options = parse_options()
    configs = parse_config(options.config, options.experiments, globals(), locals())

    # Use configuration file location as the project location.
    # Ray Tune default working directory is "~/ray_results"
    project_dir = os.path.dirname(options.config.name)
    project_dir = os.path.abspath(project_dir)

    print("Using torch version", torch.__version__)
    print("Torch device count=", torch.cuda.device_count())

    # Initialize ray cluster
    if "REDIS_ADDRESS" in os.environ:
        ray.init(redis_address=os.environ["REDIS_ADDRESS"], include_webui=True)
    else:
        # Initialize ray cluster
        ray.init(
            num_cpus=options.num_cpus,
            num_gpus=options.num_gpus,
            local_mode=options.num_cpus == 1,
        )

    # Run all experiments in parallel
    results = []
    for exp in configs:
        config = configs[exp]
        print("-" * 20)
        print(exp)
        print("-" * 20)
        config["name"] = exp
        config["num_cpus"] = options.num_cpus
        config["num_gpus"] = options.num_gpus
        config["resume"] = options.resume
        config["predict"] = options.predict
        config["plot_gradients"] = options.plot_gradients
        config["verbose"] = options.verbose

        # Make sure local directories are relative to the project location
        path = os.path.expanduser(config.get("path", "~/nta/results"))
        if not os.path.isabs(path):
            path = os.path.join(project_dir, path)
        config["path"] = path

        data_dir = os.path.expanduser(config.get("data_dir", "~/nta/datasets"))
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(project_dir, data_dir)
        config["data_dir"] = data_dir

        # When running multiple hyperparameter searches on different experiments,
        # ray.tune will run one experiment at the time. We use "ray.remote" to
        # run each tune experiment in parallel as a "remote" function and wait until
        # all experiments complete
        results.append(run_experiment.remote(config, RSMTune))

    # Wait for all experiments to complete
    ray.get(results)

    ray.shutdown()
