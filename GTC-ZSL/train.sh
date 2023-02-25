#!/bin/bash

python train.py \
  --test_data_path "./data/CLINC.json" \
  --g_file_path "./G_data/CLINCg-dataset.jsonl"\
  --label_data_path "./G_data/CLINC-labellist.json"\
  --hold_num 30\
  --class_num 10
