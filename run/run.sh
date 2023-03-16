#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# CHANGE THESE VALUES
# They were constant for the paper
# Could be better values
# Recalculate given equations in paper
# Based on upper and lower bounds
# Can see if there are other values to get the same values in paper
# Make graph of new data, and old data to compaire
# 
la=10
ua=100
lm=0.45
um=0.65
lg=45

# settings
MODEL_ARC=iresnet50
OUTPUT=./test/

mkdir -p ${OUTPUT}/vis/

python -u trainer.py \
    --arch ${MODEL_ARC} \
    --train_list /training/face-group/opensource/ms1m-112/ms1m_train.list \
    --workers 8 \
    --epochs 25 \
    --start-epoch 0 \
    --batch-size 512 \
    --embedding-size 512 \
    --last-fc-size 85742 \
    --arc-scale 64 \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --lr-drop-epoch 10 18 22 \
    --lr-drop-ratio 0.1 \
    --print-freq 100 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 1 \
    --l_a ${la} \
    --u_a ${ua} \
    --l_margin ${lm} \
    --u_margin ${um} \
    --lambda_g ${lg} \
    --vis_mag 1    2>&1 | tee ${OUTPUT}/output.log   