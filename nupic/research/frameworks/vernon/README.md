Vernon
==============
Vernon is a flexible framework to allow researchers to explore neural network models in the context of continuous, meta, and sparse learning. It is designed to be light-weight with a focus on mixin classes. 

Vernon is the creation of researchers at Numenta; note therefore that all of the caveats specified in the nupic.research README still apply. 

Getting started
==============
To get started, the key modules to be aware of are
- handlers.py : contains classes to structure the desired experiment (such as supervised vs continual learning)
- run.py and run_with_raytune.py : contain functions to run more advanced experiments
- common_models.py in our PyTorch framework : contains some common models that you can use

Running a basic experiment
==============
On your local machine you can run
```bash
python ./simple_experiment.py
```
which contains both the config for a multi-layer perceptron trained on MNIST, and the simple run function to train and evaluate it. 

User-specified parameters
==============
Given the flexibility of Vernon, there is a large number of experiment and model hyper-parameters that can be specified by the user. simple_experiment.py contains the minimal parameters to run a basic experiment, but additional parameters and their default values can be found under the main modules highlighted under 'Getting started'.

About the name
==============
Vernon is a reference to [Vernon Mountcastle](https://en.wikipedia.org/wiki/Vernon_Benjamin_Mountcastle), affectionately known as 'Mount Vernon' around Numenta, who first discovered and characterized the columnar organization of the neo-cortex.