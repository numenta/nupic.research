from models import SparseMLP, DendriticMLP
from nupic.research.frameworks.pytorch.models.common_models import StandardMLP
import torch
import time
import torch.autograd.profiler as profiler


def func(model, device, dendrite=False):
    if dendrite:
        dummy_tensor = torch.rand(1024, 11, device=device)
        dummy_context = torch.rand(1024, 10, device=device)

        s = time.time()
        with profiler.profile(record_shapes=True, use_cuda=device.type=='cuda') as prof:
            with profiler.record_function("model_inference"):
                res = model(dummy_tensor, dummy_context)
        print(res)
        print(time.time() - s)
    else:
        dummy_tensor = torch.rand(1024, 21, device=device)
        s = time.time()
        with profiler.profile(record_shapes=True, use_cuda=device.type=='cuda') as prof:
            with profiler.record_function("model_inference"):
                res = model(dummy_tensor)
        print(res)
        print(time.time() - s)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

if __name__ == '__main__':
    dendrite_net = DendriticMLP(
        hidden_sizes=(2048, 2048, 2048),
        input_size=11,
        output_dim=7,
        k_winners=False,
        relu=True,
        k_winner_percent_on=0.1,
        dim_context=10,
        num_segments=(10, 10, 10),
        sparsity=0.5,
        # dendritic_layer_class=GatingDendriticLayer
    )

    dense_net = StandardMLP(
        input_size=21,
        num_classes=7,
        hidden_sizes=(2048, 2048, 2048)
    )

    sparse_net = SparseMLP(
        input_size=21,
        output_dim=7,
        hidden_sizes=(2048, 2048, 2048),
        linear_activity_percent_on=(0.1, 0.1, 0.1),
        linear_weight_percent_on=(0.5, 0.5, 0.5),
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")


    dendrite_net.to(device)
    dense_net.to(device)
    sparse_net.to(device)

    # func(dense_net, device=device)
    # func(sparse_net, device=device)
    func(dendrite_net, device=device, dendrite=True)


