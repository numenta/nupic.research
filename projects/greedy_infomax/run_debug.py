from experiments import CONFIGS
import torch
if __name__ == '__main__':
    experiment_config = CONFIGS["small_block"]
    experiment_class = experiment_config["experiment_class"]()
    experiment_class.setup_experiment(experiment_config)

    model = experiment_class.model
    x = None
    for x, _ in experiment_class.unsupervised_loader:
        break
    # out = torch.jit.trace(model, x)
    # print(out)


    while not experiment_class.should_stop():
        experiment_class.run_epoch()