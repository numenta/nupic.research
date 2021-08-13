# Transformers

## Overview

This project is primarily concerned with exploring sparse pretrained BERT models and evaluating them using GLUE scores. Please see the results direcotry for detailed documentation on our ongoing experiments.

## How to run - single node

Local implementation using Huggingface. To run, create a new experiment dict under experiments and run using:

`python run.py <experiment_name>`

You can also run multiple experiments in sequence:

`python run.py <experiment_name_A>  <experiment_name_B>`

## How to run - multiple nodes

To run it in multiple nodes, you will require the run script from transformers-cli-utils (it can be found under ray folder in the infrastructure repository). Make sure to add transformers-cli-utils root folder to PATH after downloading.

First define the location of your cluster yaml file in `RAY_CONFIG_FILE` and the location of your AWS certification file (.pem file) in `AWS_CERT_FILE`:

`export RAY_CONFIG_FILE=<path to ray config file>`

`export AWS_CERT_FILE=<path to your AWS certification file>`

The easiest way is to add the commands that create new environment variables or modify existing ones (like PATH) to your ~/.bash_profile, so they are automatically initialized every time you open a new bash terminal.

The head and worker nodes should be the of the same type of instance, and the type selected should contain at least one GPU.
Set the variable `initial_workers` in the yaml file to initialize them all along with the head node.
Then initialize your cluster:

`ray up <path to ray config file>`

After the head and worker nodes are initialized, run using the bash script provided in transformers-cli-utils:

`run.sh <experiment_name>`

As in single node, you can run multiple experiments in sequence. When using the script with multiple experiments, wrap the experiment names in quotes so it is read as a single argument:

`run.sh "<experiment_name_A> <experiment_name_B>"`

Wait a few minutes and the output of all instances will be redirected to your local terminal.
You can follow up the experiments in the wandb link shown when training starts.

If you need to resync the files after a local change, use the sync script available in transformers-cli-utils:

`sync.sh <path to file or folder local> <path to file or folder remote>`

For any additional commands, use the remote script. For example, to verify GPU usage on all nodes, do:

`remote.sh nvidia-smi`

Or to kill all running python processes:

`remote.sh "pkill -f python"`

If required, you can reboot all instances by selecting all of them in EC2 console and selecting action > reboot from the instance drop-down menu.

### Running from the head node

There is also an option of running the multiple nodes scripts from a ray head node instead of local. In this case, the head node will play the part of local and only the worker nodes will run the commands. For this scenario it is advised to have a simple non-GPU head node, since it will only be used to issue commands to the workers.

After accessing the head node via ssh or attaching a screen, run an experiment using the same command described above: `run.sh <experiment_name>`. See transformers-cli-utils readme for more information on how to use and what modifications are required to the yaml file.

## Installing

You will require the libraries `datasets` and `transformers`.
* `datasets` can be installed using pip or from source
* install `transformers` from source, by cloning and running `pip install -e .`

The `requirement.txt` file contains a specific SHA if you want to reproduce a tested environment. We are using the latest features from these libraries and will incorporate others which are soon to be released, so for the moment those might change at a fast pace. Once we have the need to establish reproducible results we should consider more stable requirements.

## Additional notes

`transformers-cli-utils` is in a private repository. If you woud like to use it, feel free to drop me an email at lsouza at numenta dot com.
