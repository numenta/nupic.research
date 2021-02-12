# Transformers

## How to run - single node

Local implementation using Huggingface. To run, create a new experiment dict under experiments and run using:

`python run.py -e <experiment_name>`

Other accepted ways are passing arguments directly in command line or through a json file. See run.py for more details.

## How to run - multiple nodes

To run it in multiple nodes, you will require the run script from transformers-cli-utils (transformers-cli-utils is in a private repository; if you woud like to use it, feel free to drop me an email at lsouza at numenta dot com). Make sure to add transformers-cli-utils root folder to PATH after downloading.

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

### Additional notes:

transformers-cli-utils is in a private repository. If you woud like to use it, feel free to drop me an email at lsouza at numenta dot com.