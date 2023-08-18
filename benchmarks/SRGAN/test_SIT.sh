#!/bin/bash

name="UDC-SIT" # dataset name
channels=4 # UDC-SIT dataset has 4 channels
norm="norm" # UDC-SIT dataset use normalization for normalization

test_gt="data/${name}/validation/GT"
test_input="data/${name}/validation/input"
save_dir="results/${name}"
model_path="experiments/SRGAN_${name}/final/Gmodel.pth"

experiment_name="${name}_test"

# make dir to save image
mkdir -p $save_dir

python test.py \
    --name $experiment_name \
    --model-path $model_path \
    --test-GT $test_gt \
    --test-input $test_input \
    --channels $channels \
    --norm $norm
