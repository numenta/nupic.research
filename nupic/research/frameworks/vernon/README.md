Vernon
==============
Vernon is a framework for building machine learning experiments from reusable components.

Vernon embraces Python's object-oriented design. Experiments are structured as classes and are extended via subclassing. A subclass can extend an experiment directly by overriding methods, or it can use mixins via Python's multiple inheritance. Vernon encourages building up experiment classes by mixing smaller reusable classes together.

All of the caveats specified in the nupic.research README apply. 

Getting started
==============
To get started, the key modules to be aware of are
- [experiments](./experiments/) : contains classes to structure the desired experiment (such as supervised vs continual learning)
- [distributed](./distributed/) : contains extended classes that are designed to run in parallel synchronized processes
- [run.py](./run.py) and [run_with_raytune.py](./run_experiment/run_with_raytune.py) : contain functions to run more advanced experiments
- [common_models.py](../pytorch/models/common_models.py) in our PyTorch framework : contains some common models for use 

Running a basic experiment
==============
On your local machine you can run simple_experiment.py found under nupic.research/projects/vernon_examples/
```bash
python simple_experiment.py
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
Because Vernon experiments are extensible via subclassing, they automatically support mixins via Python's multiple inheritance. A subclass can override a method directly, or it can do it via mixins. An introduction to Python mixins can be found [here](https://realpython.com/inheritance-composition-python/#mixing-features-with-mixin-classes).

As an example, one could define
```python
class KWinnersSupervisedExperiment(mixins.UpdateBoostStrength,
                                   SupervisedExperiment):
    pass
```
which will run the typical SupervisedExperiment with the addition of UpdateBoostStrength. 

Looking at UpdateBoostStrength
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

Mixins have full access to the experiment class, and they can do whatever they want. Often, they will extend methods, first calling `super().method` then running additional code. Other times, they will replace methods (never calling `super().method`). Mixins can depend on the internals of specific experiments and even the internals of other mixins. Because this is research code, we don't place any firm restrictions on how mixins should work. To make the dependencies explicit and to promote reusability, as mixins become mature they should use the following practices:

- Interact with the `self` via explicit **interfaces**. Rather than having mixins interact directly with specific experiments, the experiment should implement an [interface](./interfaces/), and the mixin should interact with that interface. This enables using the mixin with any experiment exposing this interface. It also prevents errors; if any mixins have an unresolved interface dependency, the experiment class will have unimplemented abstract methods and will fail to instantiate. Finally, this enables static analysis and IDE support. An introduction to interfaces in Python can be found [here](https://realpython.com/python-interface/#python-interface-overview).
- Implement the `get_execution_order` classmethod. This documents the mixin's modifications to various stages of the experiment. The execution order list is displayed when an experiment is run.

Vernon provides a variety of mixins that enable the specification of more advanced experiments, including:
- custom loss functions (composite_loss.py)
- knowledge distillation to have the network learn from a teacher model (knowledge_distillation.py)
- efficient tuning of the learning rate (lr_range_test.py)
- MaxUp data augmentation to improve model generalization (maxup.py)

Interface edge cases
====================
As the number of interfaces increases, Vernon experiments and mixins may wrestle with Python's [MRO](https://www.python.org/download/releases/2.3/mro/). Certain combinations of mixins may not work together if they order their interfaces differently. To keep everything compatible, use these tricks:

1. If a class C inherits directly from multiple interfaces I1 and I2, it should sort those interfaces in alphabetical order. Write `class C(I1, I2)`, not `class C(I2, I1)`.
2. If a class C inherits interfaces from multiple parents (e.g. `class C(I2, D)` given `class D(I1, I3))`, it can explicitly set the order for these interfaces (e.g. `class C(D, I1, I2, I3)`).

In general, edge case issues can be fixed as they occur -- there's no need to proactively use Trick 2 on every custom experiment class.

About the name
==============
Vernon is a reference to [Vernon Mountcastle](https://en.wikipedia.org/wiki/Vernon_Benjamin_Mountcastle), who first discovered and characterized the columnar organization of the neo-cortex.
