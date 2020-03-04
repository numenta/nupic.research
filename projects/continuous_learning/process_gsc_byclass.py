import numpy as np
import torch
import sys
sys.path.append("../whydense/gsc/")

from nupic.research.frameworks.pytorch.dataset_utils import PreprocessedDataset
from sparse_speech_experiment import SparseSpeechExperiment

data_dir = "/home/ec2-user/nta/data/"

ranges = [np.arange(k,k+1) for k in range(30)]

for k in range(1,12):
    data_tensor = torch.zeros(0,1,32,32)
    for j in range(len(ranges)):
        dataset = PreprocessedDataset(cachefilepath=data_dir,
                                 basename="gsc_train",
                                 qualifiers=ranges[j])
    
        class_indices = np.where(dataset.tensors[1] == k)[0]
        if len(class_indices) > 0:
            data_tensor = torch.cat((data_tensor, torch.Tensor(dataset.tensors[0][class_indices,:,:,:])))
    labels_tensor = torch.Tensor(data_tensor.shape[0]*[k]).long()
    out_tensor = list((data_tensor, labels_tensor))
    with open("/home/ec2-user/nta/data/data_classes/data_train_{}.npz".format(k), 'wb') as f:
        torch.save(out_tensor, f)

for k in range(1,12):
    data_tensor = torch.zeros(0,1,32,32)
    dataset = PreprocessedDataset(cachefilepath=data_dir,
                             basename="gsc_valid",
                             qualifiers=[""]
                            )

    class_indices = np.where(dataset.tensors[1] == k)[0]
    
    if len(class_indices) > 0:
        data_tensor = torch.cat((data_tensor, torch.Tensor(dataset.tensors[0][class_indices,:,:,:])))
        
    labels_tensor = torch.Tensor(data_tensor.shape[0]*[k]).long()
    out_tensor = list((data_tensor, labels_tensor))
    with open("/home/ec2-user/nta/data/data_classes/data_valid.npz", 'wb') as f:
        torch.save(out_tensor, f)
    
noise_values = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for k in range(1,12):
    data_tensor = torch.zeros(0,1,32,32)
    for j in range(len(ranges)):
        dataset = PreprocessedDataset(cachefilepath=data_dir,
                                 basename="gsc_test_noise",
                                 qualifiers=["{:02d}".format(int(100*n)) for n in noise_values])
    
        class_indices = np.where(dataset.tensors[1] == k)[0]
        
        if len(class_indices) > 0:
            data_tensor = torch.cat((data_tensor, torch.Tensor(dataset.tensors[0][class_indices,:,:,:])))
    labels_tensor = torch.Tensor(data_tensor.shape[0]*[k]).long()
    out_tensor = list((data_tensor, labels_tensor))
    with open("/home/ec2-user/nta/data/data_classes/data_test_{}.npz".format(k), 'wb') as f:
        torch.save(out_tensor, f)
