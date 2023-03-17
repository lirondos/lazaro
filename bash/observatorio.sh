#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate observatory

python /home/ealvarezmellado/observatory/lazaro/scripts/observatorio.py /home/ealvarezmellado/observatory/lazaro /home/ealvarezmellado/observatory/lazaro/param/param.yaml > /home/ealvarezmellado/observatory/lazaro/logs/execution_observatory.log  2>&1 && 
python /home/ealvarezmellado/observatory/lazaro/scripts/tweet.py /home/ealvarezmellado/observatory/lazaro /home/ealvarezmellado/observatory/lazaro/param/param.yaml > /home/ealvarezmellado/observatory/lazaro/logs/execution_lazarobot.log 2>&1 