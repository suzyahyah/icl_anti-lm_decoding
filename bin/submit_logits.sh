#!/usr/bin/env bash
# Author: Suzanna Sia

#$ -l h=!r7n08*
# change this 
cd /exp/ssia/projects/icl_antilm_decoding

ml load cuda11.6/toolkit
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo $*
echo $HOSTNAME $CUDA_VISIBLE_DEVICES

echo `nvcc --version` 
seed=$1
model=$2
direction=$3
logitsp_cf=$4
nprompts=$5
alpha=$6
contrast=$7
gen_cf=$8
instr=$9

python code/run_prompts.py \
    --data.direction $direction\
    --file_paths_cfg configs/file_paths/logits.yaml\
    --data.trainset FLORES\
    --data.testset FLORES\
    --logitsp_cf $logitsp_cf\
    --model.model_size $model \
    --sample_prompts.nprompts $nprompts\
    --generator_cf $gen_cf\
    --format_cf configs/format/${instr}.yaml \
    --logitsp.contrast_logits_type $contrast \
    --generator.batch_size 16 \
    --logitsp.alpha $alpha
