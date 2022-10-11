#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/home/ljh/GEC/SCOPE
BERT_PATH=/home/ljh/GEC/SCOPE/FPT
DATA_DIR=$REPO_PATH/data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


lr=5e-5
bs=32
accumulate_grad_batches=2
epoch=30
OUTPUT_DIR=$REPO_PATH/outputs/bs${bs}epoch${epoch}

mkdir -p $OUTPUT_DIR
CUDA_VISIBLE_DEVICES=3 python -u $REPO_PATH/finetune/train.py \
--bert_path $BERT_PATH \
--data_dir $DATA_DIR \
--save_path $OUTPUT_DIR \
--max_epoch=$epoch \
--lr=$lr \
--warmup_proporation 0.1 \
--batch_size=$bs \
--gpus=0, \
--accumulate_grad_batches=$accumulate_grad_batches  \
--reload_dataloaders_every_n_epochs 1 
sleep 1

# nohup bash train.sh 2>&1 >train.log &