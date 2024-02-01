#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate observatorio

python /home/ealvarez/lazaro/scripts/observatorio.py /home/ealvarez/lazaro /home/ealvarez/lazaro/param/param.yaml > /home/ealvarez/lazaro/logs/execution_observatory.log  2>&1 && 
python /home/ealvarez/lazaro/scripts/tweet.py /home/ealvarez/lazaro /home/ealvarez/lazaro/param/param.yaml > /home/ealvarez/lazaro/logs/execution_lazarobot.log 2>&1 