#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lazaro
#rm -rfv /home/ealvarezmellado/lazarobot/embeddings_db/*
python /home/ealvarezmellado/lazaro/scripts/crf.py --include_other && python /home/ealvarezmellado/lazaro/scripts/tweet.py