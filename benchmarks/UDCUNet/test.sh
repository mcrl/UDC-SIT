#!/bin/bash

option_file=options/test/test.yml

PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
python basicsr/test.py -opt $option_file
