#!/bin/sh

cd ..

python generated_assist.py \
    --dataset=mit_indoor \
    --synset_size=1x \
    --aug \
    --aug_mode=S \
    --teacher_model=resnet50 \
    --teacher_init=lp \
    --teacher_optim=adamw \
    --student_model=mobilenetv2 \
    --distill=align_uniform \
    --train_bs=64 \
    --epochs=250 \
    --optimizer=adamw \
    --projector=mocov2 \
    --n_gpu=0
