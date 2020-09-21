Vernon
==============
Vernon is a flexible framework to allow researchers to explore neural network models under a variety of learning paradigms including continuous, meta, and sparse learning. It is designed to be light-weight with a focus on mixin classes. 

Vernon is the creation of researchers at Numenta; note therefore that all of the caveats specified in the nupic.research README still apply. 

Getting started
==============
To get started, the key modules to be aware of are
- [handlers.py](./handlers.py) : contains classes to structure the desired experiment (such as supervised vs continual learning)
- [run.py](./run_experiment/run.py) and [run_with_raytune.py](./run_experiment/run_with_raytune.py) : contain functions to run more advanced experiments
- [common_models.py](../pytorch/models/common_models.py) in our PyTorch framework : contains some common models that you can use 

Running a basic experiment
==============
On your local machine you can run simple_experiment.py found under nupic.research/projects/vernon_examples/
```bash
python ./simple_experiment.py
```
which contains both the config for a multi-layer perceptron trained on MNIST, and the simple run function to train and evaluate it. 

User-specified parameters
==============
Given the flexibility of Vernon, there is a large number of experiment and model hyper-parameters that can be specified by the user. simple_experiment.py contains the minimal parameters to run a basic experiment, but additional parameters and their default values can be found under the main modules highlighted under 'Getting started'.

Experiment class
==============
The experiment class API enables one to define a variety of requirements for training and evaluating a given model. The two main experiments provided are SupervisedExperiment and ContinualLearningExperiment. The former is the general experiment class used to train neural networks in supervised learning tasks. In brief, supervised-learning follows the pseudo-code structure of 
- configure network
- for epoch 1 to e:
    - for batch 1 to b:
        - gradient_step()
- evaluate()

In contrast, continual learning follows the below structure, with the experiment class requiring additional parameters such as **num_tasks** to reflect this:
- configure network
- for task 1 to t:
	- for epoch 1 to e:
		- for batch 1 to b:
			- gradient_step()
	- evaluate_task()
	- evaluate_all_previous_tasks()


Mixins for advanced experiments
==============
Mixins enable one to specificlaly inherit desired features from a particular parent class, without creating a rigid relationship between the parent and child. In Vernon this enables one to flexibly define new experiments. Depending on the execution order of the defined experiment, this will determine the class properties that are inherited and therefore the experiment that will be performed. 

As an example, one could define 
```python
class KWinnersSupervisedExperiment(mixins.UpdateBoostStrength,
                                         SupervisedExperiment):
    pass
```
which will run the typical SupervisedExperiment with the addition of UpdateBoostStrength. 

More specifically, looking at UpdateBoostStrength
```python
class UpdateBoostStrength:
    """
    Update the KWinners boost strength before every epoch.
    """
    def pre_epoch(self):
        super().pre_epoch()
        self.model.apply(update_boost_strength)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["pre_epoch"].append("UpdateBoostStrength")
        return eo
    pass
```
one can see that **update_boost_strength** will be applied in the parent class's pre_epoch period. 

**get_execution_order** tracks the use of mixins at various stages of the experiment, and the execution order list is displayed when an experiment is run. 

Vernon provides a variety of mixins that enable the specification of more advanced experiments, including:
- custom loss functions (composite_loss.py)
- knowledge distillation to have the network learn from a teacher model (knowledge_distillation.py)
- efficient tuning of the learning rate (lr_range_test.py)
- MaxUp data augmentation to improve model generalization (maxup.py)

About the name
==============
Vernon is a reference to [Vernon Mountcastle](https://en.wikipedia.org/wiki/Vernon_Benjamin_Mountcastle), affectionately known as 'Mount Vernon' around Numenta, who first discovered and characterized the columnar organization of the neo-cortex.