# 2019-09-30: Network Architechture Comparisons

**Do our networks perfectly mimic the How-so-dense's papers?**

## Mine:
with 
```
  net_params={
      "boost_strength": 1.5
  },
```

```
NETWORK...
gsc_sparse_dsnn:

Sequential(
  (cnn1): Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1))
  (cnn1_batchnorm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  (cnn1_kwinner): KWinners2d(channels=64, n=0, percent_on=0.095, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
  (cnn1_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (cnn2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
  (cnn2_batchnorm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  (cnn2_kwinner): KWinners2d(channels=64, n=0, percent_on=0.125, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
  (cnn2_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (flatten): Flatten()
  (linear): Linear(in_features=1600, out_features=1000, bias=True)
  (linear_bn): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
  (linear_kwinner): KWinners(n=1000, percent_on=0.1, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
  (output): Linear(in_features=1000, out_features=12, bias=True)
  (softmax): LogSoftmax()
)
```

## Lucas's
```
NETWORK...
GSCHeb:

GSCHeb(
  (features): Sequential(
    (0): Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): KWinners2d(channels=64, n=0, percent_on=0.095, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
    (4): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
    (5): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (7): KWinners2d(channels=64, n=0, percent_on=0.125, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
    (8): Flatten()
  )
  (classifier): Sequential(
    (0): DSLinearBlock(
      (0): DSLinear(in_features=1600, out_features=1500, bias=True)
      (1): BatchNorm1d(1500, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
      (2): KWinners(n=1500, percent_on=0.067, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
    )
    (1): DSLinearBlock(
      (0): DSLinear(in_features=1500, out_features=12, bias=True)
    )
  )
)
```

## Original

SuperSparse
```
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.01
    lr: 0.01
    momentum: 0.0
    nesterov: False
    weight_decay: 0.01
)
StepLR
    step_size 1
    gamma 0.9
    last_epoch -1
SparseNet(
  (cnnSdr): Sequential(
    (cnnSdr1_cnn): Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1))
    (cnnSdr1_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (cnnSdr1_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (cnnSdr1_kwinner): KWinners2d(n=12544, percent_on=0.0956632653061, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
    (cnnSdr2_cnn): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
    (cnnSdr2_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (cnnSdr2_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (cnnSdr2_kwinner): KWinners2d(n=1600, percent_on=0.125, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
  )
  (flatten): Flatten()
  (linearSdr): Sequential(
    (linearSdr1): SparseWeights(
      weight_sparsity=0.1
      (module): Linear(in_features=1600, out_features=1500, bias=True)
    )
    (linearSdr1_bn): BatchNorm1d(1500, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (linearSdr1_kwinner): KWinners(n=1500, percent_on=0.0666666666667, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
  )
  (fc): Linear(in_features=1500, out_features=12, bias=True)
  (softmax): LogSoftmax()
)

```

Sparse
```
SparseNet(
  (cnnSdr): Sequential(
    (cnnSdr1_cnn): Conv2d(1, 64, kernel_size=(5, 5), stride=(1, 1))
    (cnnSdr1_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (cnnSdr1_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (cnnSdr1_kwinner): KWinners2d(n=12544, percent_on=0.0956632653061, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
    (cnnSdr2_cnn): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
    (cnnSdr2_bn): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (cnnSdr2_maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (cnnSdr2_kwinner): KWinners2d(n=1600, percent_on=0.125, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
  )
  (flatten): Flatten()
  (linearSdr): Sequential(
    (linearSdr1): SparseWeights(
      weight_sparsity=0.4
      (module): Linear(in_features=1600, out_features=1000, bias=True)
    )
    (linearSdr1_bn): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
    (linearSdr1_kwinner): KWinners(n=1000, percent_on=0.1, boost_strength=1.5, boost_strength_factor=0.9, k_inference_factor=1.5, duty_cycle_period=1000)
  )
  (fc): Linear(in_features=1000, out_features=12, bias=True)
  (softmax): LogSoftmax()
)
```

#### Optimizer
 
```
SGD (
Parameter Group 0
    dampening: 0
    initial_lr: 0.01
    lr: 0.01
    momentum: 0.0
    nesterov: False
    weight_decay: 0.01
)
```

#### LR Scheduler
```
StepLR
    step_size 1
    gamma 0.9
    last_epoch -1
```