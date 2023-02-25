# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class DataGenerator(Dataset):
    def __init__(self, data_path, config, mode='unseen'):
        self.config = config
        self.path = data_path
        self.mode = mode

        if self.mode == 'unseen':
        # self.index_to_label = {0: 'BOOK', 1: 'PLAYLIST', 2:'MUSIC', 3:'WEATHER', 4:'MOVIE', 5:'RESTAURANT', 6:'SEARCH'}
            if config["g_file_name"] == 'SNIP7G.json':
                self.index_to_label = {0: 'BOOK', 1: 'PLAYLIST', 2:'MUSIC', 3:'WEATHER', 4:'MOVIE', 5:'RESTAURANT', 6:'SEARCH'}
                # self.index_to_label = {0: 'BOOK', 1: 'PLAYLIST'}

            elif config["g_file_name"] == 'CLINC-dataset_g.json':
                self.index_to_label = {0: 'cancel reservation', 1: 'freeze account', 2: 'current location', 3: 'how old are you', 4: 'what is your name', 5: 'bill due',
                                       6: 'exchange rate', 7: 'shopping list' }

            elif config["g_file_name"] == 'ATIS-datasetg.json':
                self.index_to_label = {0: 'distance', 1: 'flight time', 2: 'restriction',
                                       3: 'airfare', 4: 'ground service' }

            elif config["g_file_name"] == 'WOS-datasetg.json':
                self.index_to_label = {0: 'Diabetes', 1:'Birth Control', 2:'System identification', 3:'Thermodynamics',
                    4:'Headache', 5:'Manufacturing engineering', 6:'Machine design', 7:'Operational amplifier',
                    8:'Overactive Bladder', 9:'Software engineering', 10:'Allergies',
                    11:'HIV/AIDS', 12:'Skin Care', 13:'Digital control', 14:'Attention',
                    15:'Computer programming', 16:'Parenting', 17:'Problem-solving', 18:'Image processing',
                    19:'Leadership', 20:'Green Building', 21:'State space representation', 22:'Geotextile',
                    23:'Cancer', 24:'Microcontroller', 25:'Irritable Bowel Syndrome', 26:'Computer graphics',
                    27:'Children\'s Health' }
            elif config["g_file_name"] == 'Data/BANKING772-dataset.json':
                self.index_to_label = {
                    0: 'balance_not_updated_after_bank_transfer',
                    1: 'fiat_currency_support',
                    2: 'card_linking',
                    3: 'receiving_money',
                    4: 'pending_card_payment',
                    5: 'wrong_exchange_rate_for_cash_withdrawal',
                    6: 'declined_transfer',
                    7: 'order_physical_card',
                    8: 'Refund_not_showing_up',
                    9: 'verify_top_up',
                    10: 'unable_to_verify_identity',
                    11: 'get_disposable_virtual_card',
                    12: 'declined_cash_withdrawal',
                    13: 'pending_transfer',
                    14: 'apple_pay_or_google_pay',
                    15: 'topping_up_by_card',
                    16: 'card_arrival',
                    17: 'declined_card_payment',
                    18: 'pending_top_up',
                    19: 'top_up_limits',
                    20: 'top_up_by_card_charge',
                    21: 'transaction_charged_twice',
                    22: 'card_not_working',
                    23: 'exchange_charge',
                    24: 'beneficiary_not_allowed',
                    25: 'virtual_card_not_working',
                    26: 'card_payment_not_recognised',
                }
            elif config["g_file_name"] == 'Data/Liu-datasetg.json':
                self.index_to_label = {
                    0: 'movies',
                    1: 'set',
                    2: 'affirm',
                    3: 'ticket',
                    4: 'confirm',
                    5: 'events',
                    6: 'likeness',
                    7: 'sendemail',
                    8: 'order',
                    9: 'music',
                    10: 'podcasts',
                    11: 'convert',
                    12: 'game',
                    13: 'coffee',
                    14: 'volume_down',
                    15: 'volume_up',
                    16: 'quirky',
                    17: 'createoradd'
                }
            elif config["g_file_name"] == 'Data/Huffpost-datasetg.json':
                self.index_to_label = {
                0: 'TECH',
                1: 'BLACK VOICES',
                2: 'HOUSE',
                3: 'GOOD NEWS',
                4: 'HEALTHY LIVING',
                5: 'WORLD NEWS',
                6: 'BUSINESS',
                7: 'PARENTS',
                8: 'QUEER VOICES',
                9: 'GREEN',
                10: 'COLLEGE',
                11: 'ENTERTAINMENT',
                12: 'SCIENCE',
                13: 'RELIGION',
                14: 'FIFTY',
                15: 'WEIRD NEWS'
                }
            else:
                labels = []
                i = 0
                with open(self.path, encoding="utf8") as f:
                    for line in f:
                        line = json.loads(line)
                        tag = line["label"]  #
                        if tag not in labels:
                            labels.append(tag)
                self.index_to_label = {}
                for l in labels:
                    self.index_to_label[i] = l
                    i += 1

        i = 0
        labels = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["label"]  #
                if tag not in labels:
                    labels.append(tag)
        self.index_to_label = {}
        for l in labels:
            self.index_to_label[i] = l
            i += 1


        # else:
        #     if config["seen_data_path"] == "../data/Liu/01/seen_data.json":
        #         self.index_to_label = {
        #             0:'factoid',
        #               1: 'recipe',
        #         2: 'hue_lightchange',
        #         3: 'audiobook',
        #         4: 'currency',
        #         5: 'traffic',
        #         6: 'commandstop',
        #         7: 'dislikeness',
        #         8: 'negate',
        #         9: 'hue_lightdim',
        #         10: 'cleaning',
        #         11: 'radio',
        #         12: 'volume_mute',
        #         13: 'locations',
        #         14: 'wemo_on',
        #         15: 'post',
        #         16: 'hue_lightup',
        #         17: 'volume_other'
        #         }
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"])
        # self.vocab = self.load_vocab(config["vocab_path"])
        # self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["label"]
                if tag not in self.label_to_index:
                    continue
                label = self.label_to_index[tag]
                title = line["text_a"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode_plus(title, max_length=self.config["max_length"],
                                                     pad_to_max_length=True, return_attention_mask=True,return_tensors='pt')
                else:
                    input_id = self.encode_sentence(title)
                # input_id = torch.LongTensor(input_id)
                label = torch.LongTensor([label])
                self.data.append([input_id, label, line['text_a'], line['label']])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1
        return token_dict


def load_data(data_path, config, mode = 'unseen', shuffle=True):
    dg = DataGenerator(data_path, config, mode)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
