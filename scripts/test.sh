#!/bin/sh

cd ..

python ./test_dataset.py \
    --dataset=mit_indoor \
    --model=vit-b-16 \
    --model_init=lp \
    --model_opt=adamw \
    --n_gpu=0
