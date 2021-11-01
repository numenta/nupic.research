SigOpt
==============
SigOpt is an API that enables one to efficiently and easily optimize model hyper-parameters using a combination of Bayesian and other optimization methods.

Getting started
==============
The key steps to using SigOpt for your model are to:
- In your experiment config, define the sigopt_experiment_class, which enables the experiment to handle the sigopt_config and e.g. accept suggestions received from the SigOpt API
- Set-up the API variables needed by SigOpt

Defining your experiment
==============
When defining the sigopt_experiment_class, several mixins are [available](./mixins/) to help handle the hyper-parameters used by particular learning rate schedulers, such as OneCycleLR. Some example classes are provided in [common_experiments.py](.common_experiments.py).

The sigopt_config takes:
- name - the name of the experiment (e.g. 'sparse_gsc_onecyclelr_v2'); this will help you keep track of the experiments you have run, which can be viewed on the [SigOpt web interface](https://app.sigopt.com/dashboard)
- parameters - the parameters that will be optimized by SigOpt; these include user-defined bounds (see [here](https://app.sigopt.com/docs/overview/parameter_bounds) for details); note that any parameters provided here should not have a value specified in the main experiment config
- metrics - the metrics that are guiding the optimization
- parallel_bandwidth - how many hyperparameter profiles to run in parallel during the search
- observation_budget - how many observations (hyperparameter profiles) SigOpt has available to run over the course of the entire search
- project - an umbrella name to help you organize your experiments, e.g. 'sparse_gsc'

Finally, it's necessary to define the "sigopt_experiment_id" in the experiment config. Obtaining this ID is discussed below.

API Variables
==============
To run an experiment, the following variables and IDs need to be defined:
- SigOpt API key and Project key
	- These are environment variables that can be set by editing the ~/.bash_profile to contain the lines below; note that if your experiment is still unable to retrieve these when running, you may also need to set these in ~/.bashrc
 ```bash
export SIGOPT_KEY=YOUR_SIGOPT_KEY_OBTAINED_FROM_THE_SIGOPT_WEB_DASHBOARD
export SIGOPT_PROJECT=NAME_OF_YOUR_PROJECT
```
- sigopt_experiment_id
	- After setting the above environment variables and defining the experiment config, including the sigopt_config, you are ready to generate a sigopt_experiment_id (at this point it can be defined as None in the experiment config); to generate an ID, run your experiment with the --create_sigopt flag, e.g.
```bash
python run.py -e sigopt_sparse_cnn_onecyclelr --create_sigopt
```
- This will return the ID number, which can be inserted into the experiment config. At this point, the experiment can be run using e.g. the following (here using the GPU flag)
```bash
python run.py -e sigopt_sparse_cnn_onecyclelr -g 1
```