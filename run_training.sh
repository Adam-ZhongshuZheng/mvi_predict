#!/bin/bash
set -e # Exit the script if an error happens


maxeps=100
lr=1e-5
batch_size=5
flod=1

mode='A'	# P D ALL

python3 train.py  --max_epoch $maxeps --lr $lr --batch_size $batch_size -f $flod -m $mode #-r

