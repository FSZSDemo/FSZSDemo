import numpy as np
import torch
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel


class GPReviewDataset():
    def __init__(self, reviews, targets, tokenizer, max_len=160):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        encoding = tokenizer(
            review,  # 分词文本
            padding="max_length",  # padding以定义的最大长度
            max_length=self.max_len,  # 分词最大长度
            add_special_tokens=True,  # 添加特殊tokens 【cls】【sep】
            return_token_type_ids=False,  # 返回是前一句还是后一句
            return_attention_mask=True,  # 返回attention_mask
            return_tensors='pt',  # 返回pytorch类型的tensor
            truncation=True  # 若大于max则切断
        )

    return {
        'review_text': review,
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'targets': torch.tensor(target, dtype=torch.long)
    }

if __name__ == '__main__':
