#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lazarobot
cat /home/ealvarezmellado/lazaro/indices/* >> /home/ealvarezmellado/lazaro/data/articles_index.csv
rm -rfv /home/ealvarezmellado/lazaro/indices/*
#rm -rfv /home/ealvarezmellado/lazarobot/embeddings_db/*
python /home/ealvarezmellado/lazaro/scripts/crf.py --include_other True --verbose False && python /home/ealvarezmellado/lazaro/scripts/tweet.py