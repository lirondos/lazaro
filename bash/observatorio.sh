#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate observatory
#rm -rfv /home/ealvarezmellado/lazaro/logs/*
#rm -rfv /home/ealvarezmellado/lazaro/tobetweeted/*
python /home/ealvarezmellado/observatory/lazaro/scripts/observatorio.py /home/ealvarezmellado/observatory/lazaro
/home/ealvarezmellado/observatory/lazaro/param/param.yaml > /home/ealvarezmellado/observatory/lazaro/logs/execution.log  2>&1
#&& python /home/ealvarezmellado/lazaro/scripts/tweet.py