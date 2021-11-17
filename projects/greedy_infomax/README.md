# Greedy InfoMax

## Overview

This project builds off of the work of Sindy Löwe and her self-supervised Greedy 
InfoMax framework. For a more detailed description of how the core algorithm works, please see her 
paper here: [Putting An End to End-to-End: Gradient-Isolated Learning of Representations
](https://arxiv.org/abs/1905.11786).

If you would like to replicate Sindy Löwe's original experiment, please clone her 
git repository at: https://github.com/loeweX/Greedy_InfoMax

The idea behind GreedyInfoMax is to get closer to completely local learning by 
applying the [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) loss 
independently to different parts of the network. Greedily optimizing for CPC loss 
can allow networks to learn representations in an unsupervised manner through 
learning how temporally or spatially coherent parts of the input are related to each 
other. We encourage you to read the paper before proceeding further, as there are 
many relevant details not covered here and it is a reasonably straightforward read.

## How to run - single node

To run, create a new experiment dict under experiments and run using:

`python run.py <experiment_name>`

## Installing

In order to run this code, you will need both the self-supervised learning package 
and the greedy infomax package.
* install `self_supervised_learning` and `greedy_infomax` from source, by cloning and 
  running `pip install -e .`


## Additional notes



