# -*- coding: utf-8 -*-
import torch
from loader import load_data


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        self.stats_dict = {"correct":0, "wrong":0}  
    def eval(self, epoch):
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   
            with torch.no_grad():
                pred_results = self.model(input_ids) #
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc

    def write_stats(self, labels, pred_results):#
        assert len(labels) == len(pred_results)
        for true_label, pred_label in zip(labels, pred_results):
            pred_label = torch.argmax(pred_label)
            if int(true_label) == int(pred_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("--------------------")
        return correct / (correct + wrong)
