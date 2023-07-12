#!/bin/sh

python sample_stable_diff.py \
    --dataset=dtd  \
    --num_img_per_class=40 \
    --batch_size=5 \
    --n_start=0 \
    --n_gpu=0
