#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/home/ljh/GEC/SCOPE
BERT_PATH=/home/ljh/GEC/SCOPE/FPT
DATA_DIR=$REPO_PATH/data
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ckpt_path=outputs/bs32epoch30/checkpoint/epoch=25-df=80.6798-cf=78.3542.ckpt

OUTPUT_DIR=outputs/predict
mkdir -p $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=1  python -u finetune/predict.py \
  --bert_path $BERT_PATH \
  --ckpt_path $ckpt_path \
  --data_dir $DATA_DIR \
  --save_path $OUTPUT_DIR \
  --label_file $DATA_DIR/test.sighan15.lbl.tsv \
  --gpus=0,
