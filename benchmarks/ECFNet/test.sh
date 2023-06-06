#!/bin/bash

python test.py \
    --name UDCSIT-benchmark-ECFNet \
    --model-path experiments/ECFNet-3-0.5/model_latest.pth \
    --test-input data/test/input \
    --test-GT data/test/GT
