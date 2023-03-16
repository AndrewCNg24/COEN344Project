#!/usr/bin/env bash

#SBATCH --job-name=yourJobName
#SBATCH --output=detection-%j.out 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=youremail@scu.edu

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

la=10
ua=110
lm=0.45
um=0.8
lg=35
# Adjust values as needed
# la=10
# ua=100
# lm=0.45
# um=0.65
# lg=45

# settings
MODEL_ARC=iresnet50
OUTPUT=./test/

mkdir -p ${OUTPUT}/vis/

python -u trainer_dist.py \
    --arch ${MODEL_ARC} \
    --train_list yourTrainingPath/train.list \
    --workers 1 \
    --epochs 25 \
    --start-epoch 0 \
    --batch-size 256 \
    --embedding-size 256 \
    --last-fc-size 85744 \
    --learning-rate 0.1 \
    --fp16 0 \
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