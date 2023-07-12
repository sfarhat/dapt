#!/bin/sh

cd ..

python distill.py \
    --dataset=cifar100 \
    --teacher_model=resnet50 \
    --student_model=mobilenetv2 \
    --distill=align_uniform \
    --epochs=250 \
    --optimizer=adamw \
    --scheduler=none \
    --n_gpu=0
