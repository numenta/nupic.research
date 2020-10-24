# OML Implementation Specifics

Based off the repository https://github.com/khurramjaved96/mrcl

This document describes the specifics of the Online-aware Meta-learning algorithm detailed in the paper:
[Meta-Learning Representations for Continual Learning](https://arxiv.org/abs/1905.12588)

Specifically, these specifics flesh out the details for
  - meta-training: See [oml_omniglot.py](https://github.com/khurramjaved96/mrcl/blob/master/oml_omniglot.py)
  - meta-testing: See [evaluate_omniglot.py](https://github.com/khurramjaved96/mrcl/blob/master/evaluate_omniglot.py)

## Data

Dataset: Omniglot
  - 1663 character classes 
  - 20 per class
  - Split into "background" and "evaluation" sets
  - "background" set: 964 classes; labeled 1 through 963
  - "evaluation" set: 659 classes; labeled 1 through 658

Transforms:
  - Reshaped to size 84 x 84
  - No normalization on the data

One channel per image; shape will be `1 x 84 x 84` (C x W x H)

## Meta-training

Per meta-training iteration (inner loop + outer loop):
  - Three tasks are sampled (1 task == 1 class)
    - tasks sampled between `list(range(int(963/2), 963))` (that is `[481, ...,962]`)
  - Reset the output weights corresponding to the sampled classes
  - Both fast and slow data are taken over the same three sampled tasks
  - Replay data is taken from the other half of the dataset, those among classes `[0, ...,480]`

**Meta-train train (the inner loop)**
  - 5 images per task

**Meta-train test (the out loop)**
  - 5 (different) images per task (taken from the same tasks as the inner loop)
  - A remember set - 15 images sampled uniformly from the first 480 classes

## Optim (Meta-training)
  - inner_loop:
    - update_lr=0.03 (used in inner loop)
    - no explicit optimizer, but uses an SGD update rule
  - outer loop
    - meta_lr=1e-4 (used in outer loop)
    - ADAM optimizer

## The Model
Pytorch printout:
```
OMLNetwork(
  (representation): Sequential(
    (0): Conv2d(1, 256, kernel_size=(3, 3), stride=(2, 2))
    (1): ReLU()
    (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (3): ReLU()
    (4): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
    (5): ReLU()
    (6): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
    (7): ReLU()
    (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
    (9): ReLU()
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2))
    (11): ReLU()
    (12): Flatten()
  )
  (adaptation): Sequential(
    (0): Linear(in_features=2304, out_features=1000, bias=True)
  )
)

```
^ Note, the output layer has 1000 output units, which is more than needed.

  - Weight initialization:
    - bias params are set to zero
    - weights params for conv and linear layers use kaiming-normal initialization (fan-in)


### What's tranined and when
"Prediction Learning Network" == `adaption` network in above printout
"Representation Learning Network" == `representatin` network in above printout

Meta-train training:
  - All layers of the Prediction Learning Network
  - ~~Representation Learning Network (frozen)~~

Meta-train testing:
  - Both the PLN and RLN are updated

Meta-test training:
  - Prediction Learning Network
  - ~~Representation Learning Network (frozen)~~

Meta-test testing:
  - NA (nothing should be updated at the point)