from projects.greedy_infomax.experiments import CONFIGS
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.datasets import FakeData




if __name__ == '__main__':
    exp_config = CONFIGS['resnet_7_testing']
    exp = exp_config['experiment_class']()
    exp.setup_experiment(exp_config)
    exp.run_epoch()