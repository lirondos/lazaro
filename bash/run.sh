#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lazaro
cat /home/ealvarezmellado/lazaro/indices/* >> /home/ealvarezmellado/lazaro/data/articles_index.csv
rm -rfv /home/ealvarezmellado/lazaro/indices/*
#rm -rfv /home/ealvarezmellado/lazarobot/embeddings_db/*
python /home/ealvarezmellado/lazaro/scripts/crf.py --include_other > /home/ealvarezmellado/lazaro/logs/run.txt 2>&1 && python /home/ealvarezmellado/lazaro/scripts/tweet.py