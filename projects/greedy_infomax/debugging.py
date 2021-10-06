from projects.greedy_infomax.experiments import CONFIGS


if __name__ == '__main__':
    exp_config = CONFIGS["full_resnet_50"]
    exp_config["distributed"]=False
    exp_config["batches_in_epoch"]=2
    exp_config["batches_in_epoch_supervised"]=2
    exp_config["batches_in_epoch_val"]=2
    exp_config["batch_size"] = 1
    exp_config["batch_size_supervised"] = 1
    exp_config["batch_size_val"] = 1
    exp_config["supervised_training_epochs_per_validation"]=1
    exp_class = exp_config["experiment_class"]()
    exp_class.setup_experiment(exp_config)
    exp_class.run_epoch()