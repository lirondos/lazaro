#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lazarobot

N=8

mkdir /home/ealvarezmellado/lazarobot/tobepredicted/"$(date +'%d%m%Y')"
(
for newspaper in "eldiario" "elpais" "elconfidencial" "elmundo" "abc" "20minutos" "efe" "lavanguardia"; do

	((i=i%N)); ((i++==0)) && wait
	python /home/ealvarezmellado/lazarobot/rss.py --newspaper $newspaper > /home/ealvarezmellado/lazarobot/rss_logs/$newspaper.txt 2>&1 & 

done
)
