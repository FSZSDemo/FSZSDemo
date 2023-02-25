# -*- coding: utf-8 -*-



Config = {
    "model_path": "output",
    "train_data_path": "UseNewData.json",
    "valid_data_path": "WOS.json",
    # "label_data_path": "Data/Amazon4/03/amazon3.json",
    # "g_file_name": "Data/Amazon4/03/Amazon333-dataset.jsonl",
    # "label_data_path": "Data/20News/05/news205.json",
    # "g_file_name": "Data/20News/05/New2055-dataset.jsonl",
    # "label_data_path": "Data/Huffpost/04/huffpost4.json",
    # "g_file_name": "Data/Huffpost/04/huffpost444-dataset.jsonl",
    # "label_data_path": "Data/Huffpost/01/huffpost1.json",
    # "g_file_name": "Data/Huffpost/01/huffpost111-dataset.jsonl",
    # "label_data_path": "Data/Huffpost/05/huffpost5.json",
    # "g_file_name": "Data/Huffpost/05/huffpost555-dataset.jsonl",
    # "label_data_path": "Data/Amazon/02/amazon2.json",
    # "g_file_name": "Data/Amazon/02/Amazon223-dataset.jsonl",
    # "label_data_path": "Data/Amazon/05/amazon5.json",
    # "g_file_name": "Data/Amazon/05/Amazon555-dataset.jsonl",
    # "label_data_path": "Data/Amazon/04/amazon42.json",
    # "g_file_name": "Data/Amazon/04/Amazon4442-dataset.jsonl",
    # "label_data_path": "Data/Amazon/01/amazon1.json",
    # "g_file_name": "Data/Amazon/01/Amazon1112-dataset.jsonl",
    # "label_data_path": "Data/Reuters/04/reuters4.json",
    # "g_file_name": "Data/Reuters/04/Reuters4-dataset.jsonl",
    # "label_data_path": "Data/Reuters/03/reuters3.json",
    # "g_file_name": "Data/Reuters/03/Reuters3-dataset.jsonl",
    # "label_data_path": "Data/Reuters/05/reuters5.json",
    # "g_file_name": "Data/Reuters/05/Reuters5-dataset.jsonl",
    # "label_data_path": "Data/Reuters/04/reuters4.json",
    # "g_file_name": "Data/Reuters/04/Reuters4-dataset.jsonl",
    # "label_data_path": "Data/Amazon/02/amazon2.json",
    # "g_file_name": "Data/Amazon/02/Amazon222-dataset.jsonl",
    # "label_data_path": "Data/Amazon/01/amazon112.json",
    # "g_file_name": "Data/Amazon/01/Amazon113-dataset.jsonl",
    # "label_data_path": "Data/Amazon/05/amazon552.json",
    # "g_file_name": "Data/Amazon/05/Amazon553-dataset.jsonl",
    # "label_data_path": "Data/20News/05/news205.json",
    # "g_file_name": "Data/20News/05/New2055-dataset.jsonl",#"Data/Liu/02/Liu2-dataset-g.json",
    # "seen_data_path": "../data/Liu/02/train.json",
    # "label_data_path": "Data/BANKING77label_list.json",
    # "g_file_name": "Data/BANKING772-dataset.json",
    "seen_data_path": "../data/20News/01/seen_data.json",
    # "label_data_path": "Data/Huffpostlabel_list.json",
    # "g_file_name": "Data/Huffpost-datasetg.json", #"WOS-datasetg.json",
    "vocab_path": "../chars.txt",
    "model_type": "bert", 
    "max_length": 20,
    "hidden_size": 128,  
    "kernel_size": 3, 
    "num_layers": 12,
    "epoch": 20,
    "batch_size": 100,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3, 
    "pretrain_model_path": "bert-base-uncased",#r"roberta-base",  
    "seed": 1024,
    "class_num" : 8,
}