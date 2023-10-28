#!/bin/bash

TARGET_H=1792 
TARGET_W=1280
OFFSET=25

for ab in "1"; do 
    for ang in "1"; do
        python3 -u udc_preprocess.py ${TARGET_H} ${TARGET_W} ${OFFSET} ${ab} ${ang} | tee ./log_alignment_$(date '+%Y-%m-%d-%H:%M:%S').txt
    done
done
