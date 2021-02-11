# Transformers

## How to run - single node

Local implementation using Huggingface. To run, create a new experiment dict under experiments and run using:

`python run.py -e <experiment_name>`

Other accepted ways are passing arguments directly in command line or through a json file. See run.py for more details.

## How to run - multiple nodes

To run it in a cluster, first define the location of your cluster yaml file in `RAY_CONFIG_FILE` and the location of your AWS certification file (.pem file) in `AWS_CERT_FILE`:

`export RAY_CONFIG_FILE=<path to ray config file>`
`export AWS_CERT_FILE=<path to your AWS certification file>`

The easiest way is to add these commands to your ~/.bash_profile, so they are automatically initialized every time you open a new bash terminal.

The head and worker nodes should be the of the same type of instance, and the type selected should contain at least one GPU.
Set the variable `initial_workers` in the yaml file to initialize them all along with the head node.
Then initialize your cluster:

`ray up <path to ray config file>`

After the head and worker nodes are initialized, run using the bash script provided:

`./run.sh <experiment_name>`

Wait a few minutes and the output of all instances will be redirected to your local terminal.
You can follow up the experiments in the wandb link shown when training starts.

If you need to resync the files after a local change, run:

`ray up <path to ray config file> --restart-only`

In case there is any breaking change, you can reboot all instances by selecting all of them
in EC2 console and selection action> reboot.
