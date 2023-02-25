
import math
import torch
import argparse
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


from transformers import BertTokenizer, BertConfig, BertModel

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class BertEncoder(nn.Module):
    
    def __init__(self, args):
        super(BertEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(args.fileVocab, do_lower_case=True)
        config = BertConfig.from_json_file(args.fileModelConfig)   
        self.bert = BertModel.from_pretrained(args.fileModel,config=config)
    
        if args.numFreeze > 0:
            self.freeze_layers(args.numFreeze)


    def freeze_layers(self, numFreeze):
        unfreeze_layers = ["pooler"]
        for i in range(numFreeze, 12):
            unfreeze_layers.append("layer."+str(i))

        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
    
    def forward(self, text):

        tokenizer = self.tokenizer(
        text,
        padding = True,
        truncation = True,
        max_length = 250,
        return_tensors='pt'  # 返回的类型为pytorch tensor
        )
        input_ids = tokenizer['input_ids'].cuda()
        token_type_ids = tokenizer['token_type_ids'].cuda()
        attention_mask = tokenizer['attention_mask'].cuda()

        outputs = self.bert(
              input_ids,
              attention_mask=attention_mask,
              token_type_ids=token_type_ids
              )

        last_hidden_states = outputs[0]
        outputs = torch.mean(last_hidden_states, dim=1)
        
        return outputs
        
class Sampler(nn.Module):
    def __init__(self, args):
        super(Sampler, self).__init__()
        self.nway = args.numNWay
        self.kshot = args.numKShot
        self.qshot = args.numQShot
        self.dim = 768
        # TOP R
        self.k = args.k
        # the number of samples per shot
        self.num_sampled = args.sample

    def calculate_var(self, features):
        # features NK, k, 768

        v_mean = features.mean(dim=1) # NK, 768
        v_cov = []
        for i in range(features.shape[0]):
            diag = torch.var(features[i], dim=0)
            v_cov.append(diag)
        # NK, 768
        v_cov = torch.stack(v_cov)

        return v_mean, v_cov
    def forward(self, support_embddings, query_embeddings):
        # (NK, NQ)
        similarity = euclidean_dist(support_embddings, query_embeddings)
        # (N, K, NQ)
        similarity = similarity.view(self.nway, self.kshot, -1)
      
        values, indices = similarity.topk(self.k, dim=2, largest=False, sorted=True)  
        # calculate top R accuracy
        acc = []
        for i in range(self.nway):
            min_index = i * self.qshot
            max_index = (i+1) * self.qshot - 1
            for j in range(self.kshot):
                count = 0.0
                for z in range(self.k):
                    if indices[i][j][z] >= min_index and indices[i][j][z] <= max_index:
                        count += 1
                acc.append(count/(self.k + 0.0)) 
       
        acc = torch.tensor(acc)
        acc = acc.mean()
        nindices = indices.view(-1, self.k)
       
        convex_feat = []
        for i in range(nindices.shape[0]):
            convex_feat.append(query_embeddings.index_select(0, nindices[i]))
        convex_feat = torch.stack(convex_feat) # NK, k, 768
       
        # calculate variance
        convex_mean, convex_cov = self.calculate_var(convex_feat)
        n_lsamples = support_embddings.shape[0]

        sampled_data = []
        sampled_label = []
        for i in range(n_lsamples):
            mean, cov = self.distribution_calibration(support_embddings[i], convex_mean[i], convex_cov[i])
           
            samples = []
            for i in range(mean.shape[0]):
                samples.append(torch.normal(mean[i].cpu().detach(), cov[i].cpu().detach(), size=(1, self.num_sampled)))
            samples = torch.stack(samples)
            
            samples = samples.squeeze().permute(1, 0)
            sampled_data.append(samples)
           
        # NK, num_sampled, 768 -> N, K*num_smpled, 768
        sampled_data = torch.stack(sampled_data).view(self.nway, self.kshot*self.num_sampled, self.dim)
        
       
        return sampled_data, acc

    def distribution_calibration(self, query, base_mean, base_cov, alpha=0.21):
       
        calibrated_mean = (query + base_mean) / 2
        
        calibrated_cov = base_cov

        return calibrated_mean, calibrated_cov

class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.bert = BertEncoder(args)
        self.sampler = Sampler(args)

    def forward(self, text):
        support_size = self.args.numNWay * self.args.numKShot
        query_size = self.args.numNWay * self.args.numQShot
        text_embedding = self.bert(text)
      
        support_embddings = text_embedding[:support_size]  # NK X 768
        query_embeddings = text_embedding[support_size:]   # NQ X 768

        
        # N, S, dim
        sampled_data, acc = self.sampler(support_embddings, query_embeddings)
        sampled_data = sampled_data.cuda()
       

        # calculate prototypes
        c_prototypes = support_embddings.view(self.args.numNWay, -1, support_embddings.shape[1])  # N X K X dim
        
        prototypes = torch.cat((c_prototypes, sampled_data), dim=1)
        prototypes = torch.mean(prototypes, dim=1)

        # calculate original prototypes for generating loss
        original_prototypes = c_prototypes.mean(dim=1)

        
        return (prototypes, query_embeddings, acc, original_prototypes, sampled_data)


    def visual(self, text):
        support_size = self.args.numNWay * self.args.numKShot
        query_size = self.args.numNWay * self.args.numQShot
        text_embedding = self.bert(text)
      
        support_embddings = text_embedding[:support_size]  # NK X 768
        query_embeddings = text_embedding[support_size:]   # NQ X 768

        
        # N, S, dim
        sampled_data, acc = self.sampler(support_embddings, query_embeddings)
        sampled_data = sampled_data.cuda()

        c_prototypes = support_embddings.view(self.args.numNWay, -1, support_embddings.shape[1])  # N X K X dim
        
        prototypes = torch.cat((c_prototypes, sampled_data), dim=1)
        prototypes = torch.mean(prototypes, dim=1)

        
        return support_embddings, query_embeddings, sampled_data, prototypes