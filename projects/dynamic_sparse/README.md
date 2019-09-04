Introduction
==============

This repository contains the code for experiments on dynamic sparse neural networks.

TO FIX: Ray doesn't read relative paths or local modules. For now the solution to run Ray is

`PYTHONPATH=~/nta/nupic.research/projects/ python run_test.py`


Overview
==========

### Model 

A model class contains all code required to train and evaluate a predictive model. A model can contain one or more neural networks. As much as possible, particularities of each model should be agnostic to each neural network being used. Ideally, any torchvision (or related packages) default neural network could be used with a model. 

### Network

Networks are specific instances of neural networks. Can either be default, such as those loaded by torchvision.models, imported from a public available implementation, or customized for a particular task. 

### Common

Several support files, which can support one or more experiments.
- Utils: Includes Dataset class, to load datasets. Dataset loaders should be agnostic to the specific dataset being loaded, as far as possible. Also includes Trainable class, which act as interface to Ray, and more specific methods to interact with Ray Tune.

### Runs

Each run file is a different experiment, conducted at some point. Stored to keep track of past runs. New runs can be modelled based on past runs. Not part of the source code. 

### Notebooks

Tests, explorations, and analysis. Not part of the source code.

### Tests

Include all related tests implemented so far.  Broken down into:
- Unit: Regular unit tests, to verify functionalities of models, networks, and common functions
- Blackbox: Tests to evaluate if the output of a model and network are as expected. For example, a network with three 100-neurons hidden layers, trained on MNIST for 100 iterations with batch size 128, must return a validation accuracy above 90%. To allow for stochasticity, it is best if blackbox tests are averaged over multiple runs (3, 5, or more, depending on the computational complexity).
- Scripts: Free form tests, to be manually evaluated. 

### Deprecated

Code no longer being used, but which might be stored temporarily during research period.

