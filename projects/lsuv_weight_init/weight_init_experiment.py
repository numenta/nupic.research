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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
    def __init__(self):
        pass

        
class WeightInitExperiment(nn.Module):
    def __init__(self):
        super(WeightInitExperiment, self).__init__()

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

        self.model_filename = "weight_init_exp.pt"

    def model_setup(self, config):
        seed = config.get("seed", random.randint(0, 10000))
        torch.manual_seed(seed)

        self.model = Net()

        use_cuda = config.get('use_cuda')
        self.device = torch.device("cuda" if use_cuda else "cpu")

        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=config.get("batch_size"), shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=config.get("test_batch_size"), shuffle=True, **kwargs)
        self.optimizer = optim.SGD(self.parameters(), lr=config.get('learning_rate'), momentum=config.get('momentum'))

        # Flag to do LSUV
        weight_init = config.get('weight_init')

        if weight_init == 'lsuv':
            self.lsuv_weight_init(self.train_loader)
        elif weight_init == 'grassmanian':
            pass

    def model_save(self, path):
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

    def lsuv_weight_init(self, data_loader, tol_var=0.1, max_attempts=30):
        """
        LSUV algorithm from Mishkin and Matas 2015
        https://arxiv.org/abs/1511.06422
        """
        self.eval()
        self.apply(self.orthogonal_weight_init)
        data_iter = iter(data_loader)

        for idx, layer in enumerate(self.children()):
            self.apply(self.add_hook)
            data, target = next(data_iter)
            out = self.model(data)  # Ugly? Run a mini-batch
            attempts = 0
            current_sd = self.lsuv_data.get('act_dict').std()
            while abs(current_sd - 1.0) >= tol_var and (attempts < max_attempts):
                self.lsuv_data['current_coef'] = 1. / (current_sd + 1e-8)
                self.lsuv_data['correction_needed'] = True

                self.apply(self.update_weights)

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
