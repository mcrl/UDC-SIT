#!/bin/bash

dataset="UDC-SIT"
mapping="norm"
channels=4

train_input="data/${dataset}/training/input"
train_gt="data/${dataset}/training/GT"
val_input="data/${dataset}/validation/input"
val_gt="data/${dataset}/validation/GT"


exp_name="SRGAN_${dataset}"

# Step 1 saves Gmodel.pth in save_dir
# Step 2 loads the Gmodel.pth in save_dir
save_dir="experiments/${exp_name}_pretrain/final"

# Our Filesystem is slow but we have sufficient CPU memory
# We load all the data into memory to boost training speed

# python prefetch.py --prefetch-dir $train_input
# python prefetch.py --prefetch-dir $train_gt
# python prefetch.py --prefetch-dir $val_gt
# python prefetch.py --prefetch-dir $val_input

# necessary directories
mkdir -p logs
mkdir -p results

# Step 1: Pretrain generator model
python train_srresnet.py \
    --train-input $train_input \
    --train-gt $train_gt \
    --val-input $val_input \
    --val-gt $val_gt \
	--patch-size 256 \
    --batch-size 4 \
    --experiment-name "${exp_name}_pretrain" \
    --num-workers 8 \
    --mapping $mapping \
    --channels $channels \
    --img-savepath results/${exp_name}_pretrain_intermediates \
    --validation-interval 100 \
    --save-interval 100

# Step 2: Train SRGAN
python train_srgan.py \
    --train-input $train_input \
    --train-gt $train_gt \
    --val-input $val_input \
    --val-gt $val_gt \
	--patch-size 256 \
    --batch-size 4 \
    --experiment-name $exp_name \
    --num-workers 8 \
    --pretrained-model-dir $save_dir \
    --mapping $mapping \
    --channels $channels \
    --img-savepath results/${exp_name}_srgan_intermediates \
    --validation-interval 100 \
    --save-interval 100
