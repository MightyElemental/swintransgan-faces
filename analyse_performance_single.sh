#!/bin/bash

path="PATH_TO_CHECKPOINT_FOLDER"
checkpoint=$path"checkpoint-0039.pt"
evalpath=$path"eval/"

python ./trainTransGan.py \
    -W \
    --eval_path "$evalpath" \
    --ckpt_file "$checkpoint" \
    --batch_size 100 \
    --evaluate 50000 \
    --transformer_type swin

