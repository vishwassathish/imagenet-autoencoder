#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

# settings
MODEL_ARC=$1
CKPT=$2
DATASET=$3

# CUDA_LAUNCH_BLOCKING=1
python3 -u eval.py \
    --arch $MODEL_ARC \
    --val_list list/${DATASET}_list.txt \
    --workers 2 \
    --batch-size 128 \
    --print-freq 10 \
    --resume ${CKPT}

# Loss MRL > 100 epochs [0.0193, 0.0095, 0.0072, 0.0084, 0.0115]
# Loss MRL 100 epochs   [0.0213, 0.0104, 0.0074, 0.0092, 0.0136]