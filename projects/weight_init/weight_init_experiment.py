from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
from torchvision import datasets, transforms
import numpy as np

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


class WeightInitExperiment(nn.Module):
    def __init__(self):
        super(WeightInitExperiment, self).__init__()

        self.model_filename = "weight_init_exp.pt"
        self.data_dir = None

    def model_setup(self, config):
        self.data_dir = os.path.expanduser(config.get("data_dir", "data"))

        seed = config.get("seed", random.randint(0, 10000))
        torch.manual_seed(seed)
        dataset = config.get('dataset')

        self.model = TestMNIST() if dataset == 'MNIST' else TestCIFAR()

        use_cuda = config.get('use_cuda')
        self.device = torch.device("cuda" if use_cuda else "cpu")

        transforms = None
        if dataset == 'MNIST':
            transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif dataset == 'CIFAR10':
            transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        train_dataset = getattr(datasets, self.dataset)(
            self.data_dir, train=True, transform=transforms
        )
        test_dataset = getattr(datasets, self.dataset)(
            self.data_dir, train=False, transform=transforms
        )

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(train_dataset,
                           batch_size=config.get("batch_size"), 
                           shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset,
                           batch_size=config.get("test_batch_size"), 
                           shuffle=True, **kwargs)

        self.optimizer = optim.SGD(self.parameters(), lr=config.get('learning_rate'), momentum=config.get('momentum'))

        weight_init = config.get('weight_init')

        if weight_init == 'lsuv':
            initializer = LSUVWeightInit(self.model, self.train_loader)
            initializer.initialize()
        elif weight_init == 'uniform':
            pass

    def model_save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint_path = os.path.join(path, self.model_filename)

        # Use the slow method if filename ends with .pt
        if checkpoint_path.endswith(".pt"):
            torch.save(self.model, checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)


    def train_epoch(self, epoch, log_interval = 100):
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))

    def test_epoch(self):
        self.eval()
        ret = {}
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        ret['loss'] = test_loss
        return ret

