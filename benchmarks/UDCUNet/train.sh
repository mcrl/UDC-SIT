#!/bin/bash

option_file=options/train/train.yml

PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt $option_file --launcher pytorch --auto-resume
