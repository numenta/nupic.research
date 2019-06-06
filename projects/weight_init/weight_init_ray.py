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
#
# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py

import argparse
import configparser
import os

import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ray import tune
from torchvision import datasets, transforms
import random

from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model
from nupic.torch.modules import (
    Flatten,
    KWinners,
    KWinners2d,
    SparseWeights,
    rezero_weights,
    update_boost_strength,
)



class TestCIFAR(nn.Module):
    def __init__(self):
        super(TestCIFAR, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class TestMNIST(nn.Module):
    def __init__(self):
        super(TestMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class LSUVWeightInit(object):
    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader
        # Holder for parameters needed for LSUV init
        self.lsuv_data = {
            'act_dict': None,  # Output
            'hook': None, # Forward hook,
            'current_coef': None, # Mult for weights
            'layers_done': -1,
            'hook_idx': 0,
            'counter_to_apply_correction': 0,
            'correction_needed': False
        }

    def store_activations(self, module, input, output):
        # Store output of this layer on each forward pass
        self.lsuv_data['act_dict'] = output.data.cpu().numpy()

    def add_hook(self, m):
        '''
        Add forward hook to each layer
        '''
        if self.lsuv_data['hook_idx'] > self.lsuv_data['layers_done']:
            self.lsuv_data['hook'] = m.register_forward_hook(self.store_activations)
        else:
            # Done, skip
            self.lsuv_data['hook_idx'] += 1

    def update_weights(self, m):
        if not self.lsuv_data['correction_needed']:
            return
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if self.lsuv_data['counter_to_apply_correction'] < self.lsuv_data['hook_idx']:
                self.lsuv_data['counter_to_apply_correction'] += 1
            else:
                m.weight.data *= self.lsuv_data['current_coef']
                self.lsuv_data['correction_needed'] = False

    def orthogonal_weight_init(self, m):
        # Fill with semi-orthogonal matrix as per Saxe et al 2013
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)

    def initialize(self, tol_var=0.1, max_attempts=30):
        """
        Updates weights in model

        LSUV algorithm from Mishkin and Matas 2015
        https://arxiv.org/abs/1511.06422
        """
        print("Running LSUV weight initialization...")

        self.model.eval()
        self.model.apply(self.orthogonal_weight_init)
        data_iter = iter(self.data_loader)

        for idx, layer in enumerate(self.model.children()):
            self.model.apply(self.add_hook)
            data, target = next(data_iter)
            out = self.model(data)  # Ugly? Run a mini-batch
            attempts = 0
            current_sd = self.lsuv_data.get('act_dict').std()
            while abs(current_sd - 1.0) >= tol_var and (attempts < max_attempts):
                self.lsuv_data['current_coef'] = 1. / (current_sd + 1e-8)
                self.lsuv_data['correction_needed'] = True

                self.model.apply(self.update_weights)

                data, target = next(data_iter)
                out = self.model(data)
                current_sd = self.lsuv_data.get('act_dict').std()  # Repeated code?
                print('std at layer ',idx, ' = ', current_sd, 'mean = ', self.lsuv_data['act_dict'].mean())
                attempts += 1
            if attempts == max_attempts:
                print("Failed to converge after %d attempts, sd: %.3f" % (attempts, current_sd))
            else:
                print("Converged after %d attempts, sd: %.3f" % (attempts, current_sd))

            # Remove forward hook
            if self.lsuv_data['hook'] is not None:
                self.lsuv_data['hook'].remove()
            self.lsuv_data['hook'] = None

            self.lsuv_data['layers_done'] += 1
            self.lsuv_data['hook_idx'] = 0
            self.lsuv_data['counter_to_apply_correction'] = 0


class WeightInitExperiment(tune.Trainable):

    def __init__(self, config=None, logger_creator=None):
        super(WeightInitExperiment, self).__init__(config=config, logger_creator=logger_creator)
        self.model_filename = "weight_init_exp.pt"
        self.data_dir = None

    def _setup(self, config):
        self.data_dir = os.path.expanduser(config.get("data_dir", "data"))

        seed = config.get("seed", random.randint(0, 10000))
        torch.manual_seed(seed)
        dataset = config.get('dataset')

        self.model = TestMNIST() if dataset == 'MNIST' else TestCIFAR()

        use_cuda = config.get('use_cuda')
        self.device = torch.device("cuda" if use_cuda else "cpu")

        transform = None
        if dataset == 'MNIST':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif dataset == 'CIFAR10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        train_dataset = getattr(datasets, dataset)(
            self.data_dir, train=True, transform=transform
        )
        test_dataset = getattr(datasets, dataset)(
            self.data_dir, train=False, transform=transform
        )

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                           batch_size=config.get("batch_size"), 
                           shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                           batch_size=config.get("test_batch_size"), 
                           shuffle=True, **kwargs)

        self.optimizer = optim.SGD(self.model.parameters(), lr=config.get('learning_rate'), momentum=config.get('momentum'))

        weight_init = config.get('weight_init')

        if weight_init == 'lsuv':
            initializer = LSUVWeightInit(self.model, self.train_loader)
            initializer.initialize()
        elif weight_init == 'uniform':
            pass

    def _save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, self.model_filename)

        # Use the slow method if filename ends with .pt
        if checkpoint_path.endswith(".pt"):
            torch.save(self.model, checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _train(self):
        self.model.train()
        ret = {}
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self._iteration, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

        # From example
        train_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
        )

        return evaluate_model(
            model=self.model, loader=self.test_loader, device=self.device
        )

    # def _test(self):
    #     self.eval()
    #     ret = {}
    #     test_loss = 0
    #     correct = 0
    #     with torch.no_grad():
    #         for data, target in self.test_loader:
    #             data, target = data.to(self.device), target.to(self.device)
    #             output = self.model(data)
    #             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
    #             pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    #             correct += pred.eq(target.view_as(pred)).sum().item()

    #     test_loss /= len(self.test_loader.dataset)

    #     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #         test_loss, correct, len(self.test_loader.dataset),
    #         100. * correct / len(self.test_loader.dataset)))
    #     ret['loss'] = test_loss
    #     return ret

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(checkpoint_path)


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

    tune.run(
        trainable,
        name=config["name"],
        local_dir=config["path"],
        stop=stop_criteria,
        config=config,
        num_samples=config.get("repetitions", 1),
        search_alg=config.get("search_alg", None),
        scheduler=config.get("scheduler", None),
        trial_executor=config.get("trial_executor", None),
        checkpoint_at_end=config.get("checkpoint_at_end", False),
        checkpoint_freq=config.get("checkpoint_freq", 0),
        resume=config.get("resume", False),
        reuse_actors=config.get("reuse_actors", False),
        verbose=config.get("verbose", 0),
    )


def parse_config(config_file, experiments=None):
    """Parse configuration file optionally filtering for specific
    experiments/sections.

    :param config_file: Configuration file
    :param experiments: Optional list of experiments
    :return: Dictionary with the parsed configuration
    """
    cfgparser = configparser.ConfigParser()
    cfgparser.read_file(config_file)

    params = {}
    for exp in cfgparser.sections():
        if not experiments or exp in experiments:
            values = cfgparser.defaults()
            values.update(dict(cfgparser.items(exp)))
            item = {}
            for k, v in values.items():
                try:
                    item[k] = eval(v)
                except (NameError, SyntaxError):
                    item[k] = v

            params[exp] = item

    return params


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

    return optparser.parse_args()


if __name__ == "__main__":
    # Load and parse command line option and experiment configurations
    options = parse_options()
    configs = parse_config(options.config, options.experiments)

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
        config["name"] = exp
        config["num_cpus"] = options.num_cpus
        config["num_gpus"] = options.num_gpus

        # Make sure local directories are relative to the project location
        path = os.path.expanduser(config.get("path", "results"))
        if not os.path.isabs(path):
            config["path"] = os.path.join(project_dir, path)

        data_dir = os.path.expanduser(config.get("data_dir", "data"))
        if not os.path.isabs(data_dir):
            config["data_dir"] = os.path.join(project_dir, data_dir)

        # Pre-download dataset
        dataset = config.get("dataset", "CIFAR10")
        if not hasattr(datasets, dataset):
            (
                print(
                    "Dataset {} is not available in PyTorch.Please choose a "
                    "valid dataset.".format(dataset)
                )
            )
        getattr(datasets, dataset)(root=data_dir, download=True)

        # When running multiple hyperparameter searches on different experiments,
        # ray.tune will run one experiment at the time. We use "ray.remote" to
        # run each tune experiment in parallel as a "remote" function and wait until
        # all experiments complete
        results.append(run_experiment.remote(config, WeightInitExperiment))

    # Wait for all experiments to complete
    ray.get(results)

    ray.shutdown()
