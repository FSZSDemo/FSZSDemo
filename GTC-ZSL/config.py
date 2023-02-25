# -*- coding: utf-8 -*-

Config = {
    "model_path": "output",
    "train_data_path": "UseNewData.json",
    # "valid_data_path": "WOS.json",
    # "label_data_path": "labellist_WOS.json",
    # "g_file_name": "WOS-datasetg.json",
    # "valid_data_path": "./data/SNIP-data.json",
    # "label_data_path": "./data/SNIP7-labellist.json",
    # "g_file_name": "./data/SNIP7-g.json",
    "valid_data_path": "./data/SNIP0128.json",
    # # "valid_data_path": "./data/Liutest6.json",
    "label_data_path": "./data/SNIP0128-labellist.json",
    "g_file_name": "./data/SNIP0128-dataset.jsonl",
    # "valid_data_path": "./data/CLINC0128.json",
    # "valid_data_path": "./data/Liutest6.json",
    # "label_data_path": "./data/CLINC0128-labellist.json",
    # "g_file_name": "./data/CLINC0128-dataset.jsonl",
    "vocab_path": "../chars.txt",
    "model_type": "gated_cnn",  # 
    "max_length": 20,
    "hidden_size": 128,  # 
    "kernel_size": 3,  #
    "num_layers": 12,
    "epoch": 5,
    "batch_size": 100,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,  # 
    "pretrain_model_path": "bert-base-uncased" #r"roberta-base",  # 
    "seed": 1024,
    "class_num" : 8,
}
