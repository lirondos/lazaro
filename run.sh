#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lazarobot
cat /home/ealvarezmellado/lazarobot/indices/* >> /home/ealvarezmellado/lazarobot/articles_index.csv
rm -rfv /home/ealvarezmellado/lazarobot/indices/*
#rm -rfv /home/ealvarezmellado/lazarobot/embeddings_db/*
python /home/ealvarezmellado/lazarobot/baseline.py --include_other True --verbose False && python /home/ealvarezmellado/lazarobot/tweet.py > /home/ealvarezmellado/lazarobot/rss_logs/log.txt 2>&1