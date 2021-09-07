from experiments import CONFIGS
import torch
from nupic.research.frameworks.greedy_infomax.models.FullModel import WrappedSparseSmallVisionModel
from nupic.research.frameworks.greedy_infomax.utils.loss_utils import all_module_multiple_log_softmax

if __name__ == '__main__':
    experiment_config = CONFIGS["small_block"]
    experiment_config["distributed"] = False
    experiment_class = experiment_config["experiment_class"]()
    experiment_class.setup_experiment(experiment_config)

    model = experiment_class.model
    x = None
    for x, _ in experiment_class.unsupervised_loader:
        break

    model1_args = CONFIGS["max_lr_grid_search"]["model_args"]
    model1_args["sparsity"] = 0.0
    model1_args["num_channels"] = 64
    model1 = WrappedSparseSmallVisionModel(**model1_args)



    opt = experiment_class.optimizer
    #transfer weights between models
    for p, p1 in zip(model.parameters(), model1.parameters()):
        p1.data = p.data.clone()


    opt1 = torch.optim.SGD(model1.parameters(), lr=0.00425)


    torch.manual_seed(0)
    out = model(x)
    module_losses = all_module_multiple_log_softmax(out, targets=None)
    loss = module_losses.sum()
    opt.zero_grad()
    loss.backward()

    torch.manual_seed(0)
    out1 = model1(x)
    module_losses1 = all_module_multiple_log_softmax(out1, targets=None)
    loss1 = module_losses1.sum()
    opt1.zero_grad()
    loss1.backward()


    print(all([(p.data== p1.data).all() for p, p1 in zip(model.parameters(),
                                                         model1.parameters())]))

    print(all([(p.grad.data == p1.grad.data).all() for p, p1 in zip(model.parameters(),
                                                      model1.parameters())]))

    opt.step()
    opt1.step()

    print(all([(p.data == p1.data).all() for p, p1 in zip(model.parameters(),
                                                      model1.parameters())]))



