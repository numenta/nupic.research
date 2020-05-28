#!/bin/bash

sync_file() {
    rsync -av --progress -e "ssh -i ~/.ssh/ray-autoscaler_us-west-2.pem" ec2-user@$1:nta/nupic.research/projects/continuous_learning/$2 /Users/afisher/nta/nupic.research/projects/continuous_learning/$3
}

sync_file $1 $2 $3
