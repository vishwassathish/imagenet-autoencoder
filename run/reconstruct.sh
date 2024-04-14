#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python tools/reconstruct.py \
    --arch resnet18 \
    --resume /mmfs1/gscratch/rao/vsathish/matryoshka/results/caltech256-resnet18/081.pth \
    --val_list /mmfs1/gscratch/rao/vsathish/matryoshka/imagenet-autoencoder/list/caltech256_list.txt