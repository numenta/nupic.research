# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------
import time

import torch
import torch.autograd.profiler as profiler

from models import DendriticMLP, SparseMLP
from nupic.research.frameworks.pytorch.model_utils import count_nonzero_params
from nupic.research.frameworks.pytorch.models.common_models import StandardMLP


def func(model, device, input_size, dendrite=False):
    use_cuda = device.type == "cuda"
    dummy_tensor = torch.rand((1024, input_size), device=device)
    if dendrite:
        dummy_context = torch.rand((1024, model.dim_context), device=device)

        s = time.time()
        with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
            with profiler.record_function("model_inference"):
                res = model(dummy_tensor, dummy_context)
    else:
        s = time.time()
        with profiler.profile(record_shapes=True, use_cuda=use_cuda) as prof:
            with profiler.record_function("model_inference"):
                res = model(dummy_tensor)

    print("Wall clock:", time.time() - s)
    if device.type == "cuda":
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    dense_params, sparse_params = count_nonzero_params(model)
    print(f"Total params:{dense_params}, non-zero params:{sparse_params}")

    if res.sum() == 0:  # Just to make Python think we need res
        print(res.sum())


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    input_size = 10
    dim_context = 100
    output_dim = 100
    dendrite_net = DendriticMLP(
        hidden_sizes=(2048, 2048, 2048),
        input_size=input_size,
        output_dim=output_dim,
        k_winners=True,
        relu=False,
        k_winner_percent_on=0.1,
        dim_context=dim_context,
        num_segments=(10, 10, 10),
        sparsity=0.5,
        # dendritic_layer_class=GatingDendriticLayer
    ).to(device)

    dense_net = StandardMLP(
        input_size=input_size,
        num_classes=output_dim,
        hidden_sizes=(2048, 2048, 2048),
    ).to(device)

    sparse_net = SparseMLP(
        input_size=input_size,
        output_dim=output_dim,
        hidden_sizes=(2048, 2048, 2048),
        linear_activity_percent_on=(0.1, 0.1, 0.1),
        linear_weight_percent_on=(0.5, 0.5, 0.5),
        use_batch_norm=False,
    ).to(device)
    print(sparse_net)

    print("=================== DENSE NETWORK =====================")
    func(dense_net, input_size=input_size, device=device)

    print("\n\n=================== SPARSE NETWORK =====================")
    func(sparse_net, input_size=input_size, device=device)

    print("\n\n=================== SPARSE DENDRITIC NETWORK =====================")
    func(dendrite_net, input_size=input_size, device=device, dendrite=True)
