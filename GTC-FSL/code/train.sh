#!/bin/bash

python train.py \
  --dataFile "../data/HuffPost/01" \
  --generated_dataFile "../G_data/Huffpost/01/huffpost1-dataset.jsonl"\
  --label_dataFile "../G_data/Huffpost/01/huffpost1.json"\
  --fileVocab "../pre-trained-model/bert_base_uncased/vocab.txt" \
  --fileModelConfig "../pre-trained-model/bert_base_uncased/config.json" \
  --fileModel "../pre-trained-model/bert_base_uncased/pytorch_model.bin" \
  --fileModelSave "../model/huffpost_n5k1_r${name[i]}" \
  --numDevice 1 \
  --epochs 10 \
  --numNWay 5 \
  --numKShot 1 \
  --numQShot 25 \
  --episodeTrain 100 \
  --episodeTest 10 \
  --hold_num 30 \
  --label_num  16\
  --learning_rate 0.00001

