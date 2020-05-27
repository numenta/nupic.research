#!/bin/bash

sync_repo() {
    rsync -av --progress -e "ssh -i ~/.ssh/ray-autoscaler_us-west-2.pem" ec2-user@$1:nta/nupic.research/projects/continuous_learning/ /Users/afisher/nta/nupic.research/projects/continuous_learning/
    rsync -av --progress -e "ssh -i ~/.ssh/ray-autoscaler_us-west-2.pem" ec2-user@$1:nta/nupic.research/nupic/research/frameworks/pytorch/ /Users/afisher/nta/nupic.research/nupic/research/frameworks/pytorch/
    rsync -av --progress -e "ssh -i ~/.ssh/ray-autoscaler_us-west-2.pem" ec2-user@$1:nta/nupic.research/nupic/research/frameworks/continuous_learning/ /Users/afisher/nta/nupic.research/nupic/research/frameworks/continuous_learning/


}

sync_repo $1
