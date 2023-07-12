#!/bin/sh

cd ..

python train_ft.py \
    --dataset=cifar10 \
    --model=resnet50 \
    --optimizer=adamw \
    --epochs=250 \
    --n_gpu=0