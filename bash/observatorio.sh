#!/usr/bin/env bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lazaro
#rm -rfv /home/ealvarezmellado/lazaro/logs/*
#rm -rfv /home/ealvarezmellado/lazaro/tobetweeted/*
python /home/ealvarezmellado/lazaro/scripts/observatorio.py /home/ealvarezmellado/lazaro
/home/ealvarezmellado/lazaro/param/param.yaml > /home/ealvarezmellado/lazaro/logs/log.txt  2>&1
#&& python /home/ealvarezmellado/lazaro/scripts/tweet.py