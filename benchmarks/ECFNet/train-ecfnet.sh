#!/bin/bash

loss_lambda=$1
if [[ -z $loss_lambda ]]; then
    echo "Usage: $0 <loss_lambda>"
    exit 1
fi

echo "$0 loss lambda $loss_lambda started Phase 1"

train_input="data/training/input"
train_gt="data/training/GT"
val_input="data/validation/input"
val_gt="data/validation/GT"

step_1="ECFNet-1-${loss_lambda}"
step_2="ECFNet-2-${loss_lambda}"
step_3="ECFNet-3-${loss_lambda}"

python train.py \
    --train-input $train_input \
    --train-gt $train_gt \
    --val-input $val_input \
    --val-gt $val_gt \
	--patch-size 256 \
    --batch-size 4 \
    --loss-lambda $loss_lambda \
    --experiment-name $step_1 \
    --auto-resume \
    --num-workers 8 \
    --num-epochs 1000 \
    --validation-interval 100 \
    --save-interval 100 \
    --lr 0.0002

if [[ $? -ne 0 ]]; then
    echo "############################################################"
    echo "$0 loss lambda $loss_lambda Phase 1 failed"
    echo "############################################################"
    exit 1
fi

echo "$0 loss lambda $loss_lambda Phase 2 started"


python train.py \
    --train-input $train_input \
    --train-gt $train_gt \
    --val-input $val_input \
    --val-gt $val_gt \
	--patch-size 512 \
    --batch-size 1 \
    --loss-lambda $loss_lambda \
    --experiment-name $step_2 \
    --auto-resume \
    --num-workers 8 \
    --num-epochs 300 \
    --validation-interval 30 \
    --save-interval 30 \
    --lr 0.00001 \
    --pretrained-model experiments/$step_1/model_latest.pth

if [[ $? -ne 0 ]]; then
    echo "############################################################"
    echo "$0 loss lambda $loss_lambda Phase 2 failed"
    echo "############################################################"
    exit 1
fi

echo "$0 loss lambda $loss_lambda Phase 3 started"

python train.py \
    --train-input $train_input \
    --train-gt $train_gt \
    --val-input $val_input \
    --val-gt $val_gt \
	--patch-size 800 \
    --batch-size 1 \
    --loss-lambda $loss_lambda \
    --experiment-name $step_3 \
    --auto-resume \
    --num-workers 8 \
    --num-epochs 150 \
    --validation-interval 15 \
    --save-interval 15 \
    --lr 0.000008 \
    --aggressive-checkpointing \
    --pretrained-model experiments/$step_2/model_latest.pth

if [[ $? -ne 0 ]]; then
    echo "############################################################"
    echo "$0 loss lambda $loss_lambda Phase 3 failed"
    echo "############################################################"
    exit 1
fi

echo "$0 loss lambda $loss_lambda finished"

