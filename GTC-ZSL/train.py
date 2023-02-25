# -*- coding: utf-8 -*-
import torch
import os
import random
import os
import numpy as np
import logging
import argparse
from config import Config
from model import TorchModel, choose_optimizer
from model2 import TorchModel2, choose_optimizer
from evaluate import Evaluator
from loader import load_data
import torch.nn.functional as F
from transformers import BertModel
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score
from B_Classifier import B_Classifier,B_ClassifierIntentPredictor
import torch.nn as nn
import json
# [DEBUG,INFO,WARNING,ERROR,CRITICAL]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


seed = 1 #Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# return_dict=False

def getpair(pair_tensor, sample1_tensor, compare_tensor):
    sample_len = sample1_tensor.size()[0]
    pair_sample_len = compare_tensor.size()[0]
    pair_tensor[:sample_len] = sample1_tensor
    pair_tensor[sample_len:sample_len + pair_sample_len] = compare_tensor
    return  pair_tensor

def loss_with_label_smoothing(label_ids, logits, label_distribution, coeff=0.1):
    # label smoothing
    label_ids = label_ids.cpu()
    target_distribution = torch.FloatTensor(logits.size()).zero_()
    for i in range(label_ids.size(0)):
        target_distribution[i, label_ids[i]] = 1.0
    target_distribution = coeff * label_distribution.unsqueeze(0) + (1.0 - coeff) * target_distribution.cuda()

    # KL-div loss
    prediction = torch.log(torch.softmax(logits, dim=1))
    loss = F.kl_div(prediction, target_distribution, reduction='mean')

    return loss

def get_optimizer(model,learnrate):

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    t_total = 100
    warmup_proportion = 0.1
    optimizer = AdamW(optimizer_grouped_parameters, lr=learnrate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * warmup_proportion),
                                                num_training_steps=t_total)

    return optimizer, scheduler


class Net(nn.Module):
    def __init__(self,sample_num, label_num):
        super(Net, self).__init__()
        self.hidden_size = 100
        self.out_size = label_num #7 #2
        self.fc1 = nn.Linear(sample_num, self.hidden_size).cuda()
        self.fc2 = nn.Linear(self.hidden_size, self.out_size).cuda()
        self.dropout = nn.Dropout(0.5).cuda()
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x)
        return x


class InputExample(object):

    def __init__(self, text_a, text_b, label = None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def nli_model(input_ids, mask_id, token_type_ids, label_ids, label_single):
    # model =  BertModel.from_pretrained("bert-base-uncased").cuda()
    bert_model = 'roberta-base'
    state_dict = 'pytorch_model.bin'
    config = AutoConfig.from_pretrained(bert_model, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base',state_dict=torch.load('./pytorch_model.bin'), config=config).cuda()
    sample_num = len(input_ids)
    gradient_accumulation_steps = 2
    max_grad_norm = 1
    step = 0
    optimizer, scheduler = get_optimizer(model,1e-5)
    idx = [i for i in range(0,sample_num)]
    random.shuffle(idx)
    input_ids_part1 = torch.stack(input_ids)[idx]
    mask_id_part1 = torch.stack(mask_id)[idx]
    token_type_ids_part1 = torch.stack(token_type_ids)[idx]
    label_ids_part1 = torch.stack(label_ids)[idx]
    epoch_size = 10
    for epoch in range(0,epoch_size):
        for i in range(0,sample_num):
            qian = True
            if qian==True:
                p_idx = torch.tensor([k for k in range(0, label_ids_part1[i].size(0)) if label_ids_part1[i][k] == 0])
                n_idx = torch.tensor([k for k in range(0, label_ids_part1[i].size(0)) if label_ids_part1[i][k] != 0])
                n_idx_idx = torch.multinomial(n_idx.float(),p_idx.size(0))
                n_idx_final = n_idx[n_idx_idx]
                input_id_n = input_ids_part1[i][n_idx_final]
                input_id_p = input_ids_part1[i][p_idx]
                mask_id_n = mask_id_part1[i][n_idx_final]
                mask_id_p = mask_id_part1[i][p_idx]
                token_type_id_n = token_type_ids_part1[i][n_idx_final]
                token_type_id_p = token_type_ids_part1[i][p_idx]
                label_n = label_ids_part1[i][n_idx_final]
                label_p = label_ids_part1[i][p_idx]
                input_id_final = torch.cat((input_id_n,input_id_p),dim=0)
                mask_id_final = torch.cat((mask_id_n,mask_id_p),dim=0)
                token_type_id_final = torch.cat((token_type_id_n,token_type_id_p),dim=0)
                label_final = torch.cat((label_n,label_p),dim=0)
                nli_num = input_id_final.size(0)
                idx = [i for i in range(0, nli_num)]
                random.shuffle(idx)
                input_id_final = input_id_final[idx]
                mask_id_final = mask_id_final[idx]
                token_type_id_final = token_type_id_final[idx]
                label_final = label_final[idx]

                output = model(input_ids = input_id_final.long(),attention_mask = mask_id_final.long(),token_type_ids = token_type_id_final.long())
                nli_out = output[0]
                _, num = torch.unique(label_final, return_counts=True)
                label_distribution = num/num.sum()
                loss = loss_with_label_smoothing(label_final.long(), nli_out, label_distribution)
            else:
                output = model(input_ids = input_ids_part1[i].long(),attention_mask = mask_id_part1[i].long(),token_type_ids = token_type_ids_part1[i].long())
                nli_out = output[0]
                _, num = torch.unique(label_ids_part1[i], return_counts=True)
                label_distribution = num/num.sum()
                loss = loss_with_label_smoothing(label_ids_part1[i].long(), nli_out, label_distribution)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps
            print(loss)
            loss.backward()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
            step += 1

    similar_matrix = torch.zeros(sample_num,sample_num).cuda()
    for i in range(0,sample_num):
        with torch.no_grad():
            output = model(input_ids = input_ids[i].long(),attention_mask = mask_id[i].long(),token_type_ids = token_type_ids[i].long())

            # use_out = output[0][:,0]
            softmax = nn.Softmax(dim=1)
            o = softmax(output[0])
            use_out = o[:,0] #[:, 0]
        similar_matrix[i] = use_out
    epoch_part2 = 100
    label_num = 5
    classification = Net(sample_num, label_num)
    # optimizer_part2, scheduler_part2 = get_optimizer(classification ,learnrate = 0.01)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in classification.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {"params": [p for n, p in classification.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer_part2 = AdamW(optimizer_grouped_parameters, lr=1e-2, eps=1e-8)
    crossentropyloss = nn.CrossEntropyLoss()
    for epoch2 in range(0,epoch_part2):
        model.eval()
        classification.train()
        idx = [i for i in range(0, sample_num)]
        random.shuffle(idx)
        input_x = similar_matrix[idx]
        input_labels = label_single[idx]
        outputx = classification(input_x)
        loss2 = crossentropyloss(outputx,input_labels)
        print(loss2)
        loss2.backward()
        optimizer_part2.step()
        # scheduler_part2.step()
        classification.zero_grad()
    return model, classification

def parse_sklearn_log(log, digits=4, mode='micro'):
    """
    :param log: skleran.precision_recall_fscore_support output (average=None)
    :param split_point: int, split to seen and unseen group
    :param digits: int, decimal limits
    :param mode: (micro | macro)
    :return: {metric:value}
    """

    support = log[3]
    precision = log[0]
    recall = log[1]
    f1 = log[2]

    if mode == 'macro':
        count_seen = split_point
        count_unseen = support.shape[0]-split_point
        count_all = support.shape[0]
    else:
        precision = precision * support
        recall = recall * support
        f1 = f1 * support
        count_unseen = support[:].sum()
        count_all = support.sum()

    perform = dict(
        pre_unseen=precision[:].sum()/count_unseen,
        rec_unseen=recall[:].sum()/count_unseen,
        f1_unseen=f1[:].sum()/count_unseen,
        pre_all=precision.sum()/count_all,
        rec_all=recall.sum()/count_all,
        f1_all=f1.sum()/count_all,
    )

    perform = {(mode + '_' + k): round(v, digits) for k, v in perform.items()}

    return perform


def test(test_data,input_x_hold,label_data,model_encode,model_classification):
    model_encode.eval()
    model_classification.eval()
    compare_num = torch.tensor([len(input_x_hold['input_ids'][i]) for i in input_x_hold['input_ids']]).sum()
    target_list = []
    pred_list = []
    type_id = False
    for index, batch_data in enumerate(test_data):
        input_ids, labels = batch_data
        if type_id==True:
            samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                       'attention_mask': input_ids['attention_mask'].squeeze(1),
                       'token_type_ids': input_ids['token_type_ids'].squeeze(1)}
        else:
            samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                       'attention_mask': input_ids['attention_mask'].squeeze(1)}
        test_matrix = []
        for sample_id in range(0,samples['input_ids'].size()[0]):
            # samples[sample] = samples[sample].cuda()
            # _, num = torch.unique(samples['input_ids'][0], return_counts=True)
            a, num = torch.unique(samples['input_ids'][sample_id], return_counts=True)
            if 0 in a:
                num[0] = num[0]
            else:
                num[0] = 0
            sample_num = 20 - num[0]
            sample_id_tensor = torch.zeros(compare_num, 40).cuda()
            sample_mask_id_tensor = torch.zeros(compare_num, 40).cuda()
            sample_token_id_tensor = torch.zeros(compare_num, 40).cuda()
            sample_input = {}
            sample_id_tensor[:,:20] = samples['input_ids'][sample_id].unsqueeze(0).repeat(compare_num,1).cuda()
            sample_mask_id_tensor[:, :20] = samples['attention_mask'][sample_id].unsqueeze(0).repeat(compare_num, 1).cuda()
            if type_id==True:
                sample_token_id_tensor[:, :20] = samples['token_type_ids'][sample_id].unsqueeze(0).repeat(compare_num, 1).cuda()
            tensor_id = 0
            label_size = len(input_x_hold['input_ids'])
            for pair_label in range(0, label_size):
                for pair_id in range(0, len(input_x_hold['input_ids'][pair_label])):
                    # _, num = torch.unique(input_x_hold['input_ids'][pair_label][pair_id], return_counts=True)
                    a, num = torch.unique(input_x_hold['input_ids'][pair_label][pair_id], return_counts=True)
                    if 0 in a:
                        num[0] = num[0]
                    else:
                        num[0] = 0
                    compare_sample_num = 20 - num[0] - 1
                    sample_id_tensor[tensor_id][sample_num:sample_num + compare_sample_num] = input_x_hold['input_ids'][pair_label][pair_id][1:compare_sample_num + 1]
                    sample_mask_id_tensor[tensor_id][sample_num:sample_num + compare_sample_num] = \
                    input_x_hold['attention_mask'][pair_label][pair_id][1:compare_sample_num + 1]
                    if type_id==True:
                        sample_token_id_tensor[tensor_id][sample_num:sample_num + compare_sample_num] = \
                        input_x_hold['token_type_ids'][pair_label][pair_id][1:compare_sample_num + 1]
                    tensor_id += 1
            # print("here")
            sample_input['input_ids'] = sample_id_tensor
            sample_input['attention_mask'] = sample_mask_id_tensor
            if type_id==True:
                sample_input['token_type_ids'] = sample_token_id_tensor
            test_matrix.append(sample_input)
        with torch.no_grad():
            test_batch_matrix = []
            for sample_input in test_matrix:
                if type_id==True:
                    output = model_encode(input_ids=sample_input['input_ids'].long(), attention_mask=sample_input['attention_mask'].long(),
                                   token_type_ids=sample_input['token_type_ids'].long())
                else:
                    output = model_encode(input_ids=sample_input['input_ids'].long(), attention_mask=sample_input['attention_mask'].long())
                # nli_out = output[0]
                softmax = nn.Softmax(dim=1)
                o = softmax(output[0])
                test_nli_output = o[:,0]
                # test_nli_output = nli_out[:,0]
                test_batch_matrix.append(test_nli_output)
            test_batch_matrix = torch.stack(test_batch_matrix)
            test_output = model_classification(test_batch_matrix)
            target = batch_data[1].squeeze(1)
            _, pred = test_output.max(1)
            target_list.append(target)
            pred_list.append(pred)
            print(precision_score(target.cpu(), pred.cpu(), average='weighted'))
    # target_list = torch.stack(target_list)
    # pred_list = torch.stack(pred_list)
    flag = 0
    for i in target_list:
        if flag == 0:
            target_temp = i
            flag = 1
        else:
            target_temp = torch.cat((target_temp, i))
    flag = 0
    for i in pred_list:
        if flag == 0:
            pred_temp = i
            flag = 1
        else:
            pred_temp = torch.cat((pred_temp, i))

    # print(precision_score(target_list.cpu(), pred_list.cpu(), average='weighted'))
    print(precision_score(target_temp.cpu(), pred_temp.cpu(), average='weighted'))
    from sklearn.metrics import precision_recall_fscore_support
    p = precision_recall_fscore_support(target_temp.cpu(), pred_temp.cpu())
    plf = parse_sklearn_log(p)
    print(plf)
    # print("here")




def main(config):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path',
                        type=str,
                        help='path to dataset')
    parser.add_argument('--label_data_path',
                        type=str,
                        help='path to labels')
    parser.add_argument('--g_file_path',
                        type=str,
                        help='path to generated data')
    parser.add_argument('--hold_num',
                        type=int,
                        help='num of anchors per class ',
                        default=30)
    parser.add_argument('--class_num',
                        type=int,
                        help='num of unseen classes ',
                        default=10)
    parser.add_argument("--seed",
                        default=1, #2023,#42
                        type=int,
                        help="Random seed")
    parser.add_argument("--bert_model",
                        default='bert-base-uncased',
                        type=str,
                        help="BERT model")
    parser.add_argument("--train_batch_size",
                        default=370,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=30,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False, default=1.0,
                        type=float)
    parser.add_argument('--label_smoothing',
                        type=float,
                        default=0.1,
                        help='Coefficient for label smoothing (default: 0.1, if 0.0, no label smoothing)')
    parser.add_argument('--max_seq_length',
                        type=int,
                        default=128,
                        help='Maximum number of paraphrases for each sentence')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lowercase input string")

    # Special params
    parser.add_argument('--train_file_path',
                        type=str,
                        default=None,
                        help='Training data path')
    parser.add_argument('--dev_file_path',
                        type=str,
                        default=None,
                        help='Validation data path')
    parser.add_argument('--oos_dev_file_path',
                        type=str,
                        default=None,
                        help='Out-of-Scope validation data path')

    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help='Output file path')
    parser.add_argument('--save_model_path',
                        type=str,
                        default='',
                        help='path to save the model checkpoints')

    parser.add_argument('--bert_nli_path',
                        type=str,
                        default='',
                        help='The bert checkpoints which are fine-tuned with NLI datasets')

    parser.add_argument("--scratch",
                        action='store_true',
                        help="Whether to start from the original BERT")

    parser.add_argument('--over_sampling',
                        type=int,
                        default=0,
                        help='Over-sampling positive examples as there are more negative examples')

    parser.add_argument('--few_shot_num',
                        type=int,
                        default=5,
                        help='Number of training examples for each class')
    parser.add_argument('--num_trials',
                        type=int,
                        default=10,
                        help='Number of trials to see robustness')

    parser.add_argument("--do_predict",
                        action='store_true',
                        default=False,
                        help="do_predict the model")
    parser.add_argument("--do_final_test",
                        action='store_true',
                        help="do_predict the model")

    args = parser.parse_args()
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
        parser.add_argument('--test_data_path',
                            type=str,
                            help='path to dataset')
        parser.add_argument('--label_data_path',
                            type=str,
                            help='path to labels')
        parser.add_argument('--g_file_path',
                            type=str,
                            help='path to generated data')
    config["valid_data_path"] = args.test_data_path
    # # "valid_data_path": "./data/Liutest6.json",
    config["label_data_path"] = args.label_data_path
    config["g_file_name"] = args.g_file_path

    top_data = []
    # with open('./TOP-dataset7.json', 'r', encoding='utf8') as file:
    #     for line in file.readlines():
    #         dic = json.loads(line)
    #         top_data.append(dic)
    #     print("here")
    file = open(config["g_file_name"], 'r', encoding='utf-8')
    content = []
    for line in file.readlines():
        dict = json.loads(line)
        if len(dict['text_a'].split()) > 3 and dict not in content:
            content.append(dict)
    with open("./UseNewData.json", "w") as f:
        for i in content:
            line = json.dumps(i, ensure_ascii=False)
            f.write(line + '\n')

    train_data = load_data(config["train_data_path"], config)
    label_data = load_data(config["label_data_path"], config)
    test_data = load_data(config['valid_data_path'],config)



    p1 = []
    p2 = []
    p3 = []
    p4 = []
    for epoch in range(config["epoch"]):
        model = TorchModel(config)
        model2 = TorchModel2(config)
        cuda_flag = torch.cuda.is_available()
        # cuda_flag = False#
        if cuda_flag:
            model = model.cuda()
        hold_num = args.hold_num
        label_num = args.class_num #10 #10 #2  # 17 #7 #28
        criterion = nn.CrossEntropyLoss()
        no_decay = ['bias', 'LayerNorm.weight']
        weight_decay = 1e-2
        learning_rate = 1e-5
        type_id = False

        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # optimizer = AdamW(model.parameters(), lr=learning_rate)
        optimizer1 = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        output_x = {}
        input_x = {}
        input_x['input_ids'] = {}
        input_x['attention_mask'] = {}
        input_x['token_type_ids'] = {}
        input_x['text'] = {}
        input_x['text_label'] = {}
        input_x['score'] = {}

        input_x['emb'] = {}
        # model.eval()
        index = 0
        for index, batch_data in  enumerate(label_data):
            input_ids, labels, text, text_label = batch_data
            # index += 1
            if type_id == True:
                samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                           'attention_mask': input_ids['attention_mask'].squeeze(1),
                           'token_type_ids': input_ids['token_type_ids'].squeeze(1)}
            else:
                samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                           'attention_mask': input_ids['attention_mask'].squeeze(1)}
            for sample in samples:
                samples[sample] = samples[sample].cuda()
            label_output = model(samples,labels)
            # criterion(label_data,batch_data[1])
            # loss = criterion(label_output[0],batch_data[1].squeeze(1).cuda())
            # loss.backward()
            # print(loss)
            # optimizer1.step()
            # label_output = torch.zeros(len(labels),768).cuda()
            # ed_temp = []
            # ed_all = []
            # for i in range(0,label_data.size()[0]):
            #     label_output[labels[i]] = label_data[i]
            #     ed_temp.append(0)
        input_x_hold = {}
        input_x_hold['input_ids'] = {}
        input_x_hold['attention_mask'] = {}
        input_x_hold['token_type_ids'] = {}
        input_x_hold['text'] = {}
        input_x_hold['text_label'] = {}
        input_x_hold['score'] = {}
        label_size = label_output[0].size(0)
        # optimizer1, scheduler1 = get_optimizer(model, 1e-3)
        for label in range(0,label_size):
            output_x[label] = []
            # input_x[label] = []
            input_x['input_ids'][label]  = []
            input_x['attention_mask'][label]  = []
            input_x['token_type_ids'][label]  = []
            input_x['text'][label] = []
            input_x['text_label'][label] = []
            input_x['score'][label] = []
            input_x['emb'][label] = []
            input_x_hold['input_ids'][label]  = []
            input_x_hold['attention_mask'][label]  = []
            input_x_hold['token_type_ids'][label]  = []
            input_x_hold['text'][label] = []
            input_x_hold['text_label'][label] = []
            input_x_hold['score'][label] = []
        model.train()
        epoch_num = 5
        BERT_NLI_PATH = None #'pytorch_model.bin'
        model_part2 = B_Classifier(path=BERT_NLI_PATH,args=args)
        # model_part3 = Net(hold_num * label_num, label_num)
        for i in range(0,5):
            # index = 0
            for index, batch_data in enumerate(train_data):  #
            # for batch_data in train_data.dataset.data:
                # batch_data = [d.cuda() for d in batch_data]
                optimizer1.zero_grad()
                input_ids, labels, text, label_text = batch_data
                if type_id == True:
                    samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                               'attention_mask': input_ids['attention_mask'].squeeze(1),
                               'token_type_ids': input_ids['token_type_ids'].squeeze(1)}
                else:
                    samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                               'attention_mask': input_ids['attention_mask'].squeeze(1)}

                for sample in samples:
                    samples[sample] = samples[sample].cuda()
                # with torch.no_grad():
                #     x = model(samples, labels)
                x = model(samples, labels)
                loss = criterion(x[0], batch_data[1].squeeze(1).cuda())
                y_pred_label = x[0].argmax(dim=1)
                acc = ((y_pred_label == labels.view(-1).cuda()).sum()).item()
                # loss = criterion(y_pred_prob.view(-1, 2), label.view(-1))
                loss.backward()
                l = loss.item()
                print( str(l) +"----"+str(acc))
                optimizer1.step()
            label_size = label_output[0].size(0)
        allacc = 0
        label_target = []
        pred_list = []
        for index, batch_data in enumerate(test_data):
            with torch.no_grad():
                input_ids, labels, text, label_text = batch_data
                if type_id == True:
                    samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                               'attention_mask': input_ids['attention_mask'].squeeze(1),
                               'token_type_ids': input_ids['token_type_ids'].squeeze(1)}
                else:
                    samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                               'attention_mask': input_ids['attention_mask'].squeeze(1)}

                for sample in samples:
                    samples[sample] = samples[sample].cuda()
                # with torch.no_grad():
                #     x = model(samples, labels)
                x = model(samples, labels)
                y_pred_label = x[0].argmax(dim=1)
                acc = ((y_pred_label == labels.view(-1).cuda()).sum()).item()
                label_target.extend(labels.view(-1).cuda().tolist())
                pred_list.extend(y_pred_label.cuda().tolist())
                allacc += acc
        # index = 0
        print(allacc)
        from sklearn.metrics import precision_recall_fscore_support
        p = precision_recall_fscore_support(torch.tensor(label_target), torch.tensor(pred_list))
        plf_finetune = parse_sklearn_log(p)
        print(plf_finetune)
        p4.append(plf_finetune)
        if_topk = False
        use_bert_fit = False
        if use_bert_fit == True:
            for index , batch_data in enumerate(train_data):
                # index += 1
                input_ids, labels, text, label_text = batch_data
                model.eval()
                if type_id == True:
                    samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                               'attention_mask': input_ids['attention_mask'].squeeze(1),
                               'token_type_ids': input_ids['token_type_ids'].squeeze(1)}
                else:
                    samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                               'attention_mask': input_ids['attention_mask'].squeeze(1)}

                for sample in samples:
                    samples[sample] = samples[sample].cuda()
                # with torch.no_grad():
                #     x = model(samples, labels)
                with torch.no_grad():
                    x = model(samples, labels)
                    sample_num = x[0].size(0)
                    pred_y = x[0].argmax(1)
                    target_label = labels.view(-1)
                    softmax = nn.Softmax(dim=0)
                    x_softmax = softmax(x[0])
                    print("here")
                    for i in range(0,sample_num):
                        if pred_y[i] != target_label[i]:
                            continue
                        input_x['input_ids'][labels[i].item()].append(samples['input_ids'][i])
                        input_x['attention_mask'][labels[i].item()].append(samples['attention_mask'][i])
                        if type_id==True:
                            input_x['token_type_ids'][labels[i].item()].append(samples['token_type_ids'][i])
                        input_x['text'][labels[i].item()].append(text[i])
                        input_x['text_label'][labels[i].item()].append(label_text[i])
                        input_x['score'][labels[i].item()].append(x_softmax[i][labels[i].item()])
                print("here")
            label_score = []
            for i in range(0,label_size):
                label_score = torch.tensor(input_x['score'][i])
                idx = torch.multinomial(label_score, hold_num, replacement=False)
                for k in idx:
                    input_x_hold['input_ids'][i].append(input_x['input_ids'][i][k])
                    input_x_hold['attention_mask'][i].append(input_x['attention_mask'][i][k])
                    if type_id == True:
                        input_x_hold['token_type_ids'][i].append(input_x['token_type_ids'][i][k])
                    input_x_hold['text'][i].append(input_x['text'][i][k])
                    input_x_hold['text_label'][i].append(input_x['text_label'][i][k])
        else:
            # input_x['emb'] = []
            for index, batch_data in enumerate(train_data):
                # index += 1
                input_ids, labels, text, label_text = batch_data
                model.eval()
                if type_id == True:
                    samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                               'attention_mask': input_ids['attention_mask'].squeeze(1),
                               'token_type_ids': input_ids['token_type_ids'].squeeze(1)}
                else:
                    samples = {'input_ids': input_ids['input_ids'].squeeze(1),
                               'attention_mask': input_ids['attention_mask'].squeeze(1)}

                for sample in samples:
                    samples[sample] = samples[sample].cuda()
                # with torch.no_grad():
                #     x = model(samples, labels)
                with torch.no_grad():
                    x = model2(samples, labels)
                    # print("here")
                    sample_num = x.size(0)
                    for i in range(0, sample_num):
                        input_x['input_ids'][labels[i].item()].append(samples['input_ids'][i])
                        input_x['attention_mask'][labels[i].item()].append(samples['attention_mask'][i])
                        if type_id == True:
                            input_x['token_type_ids'][labels[i].item()].append(samples['token_type_ids'][i])
                        input_x['text'][labels[i].item()].append(text[i])
                        input_x['text_label'][labels[i].item()].append(label_text[i])
                        input_x['emb'][labels[i].item()].append(x[i].tolist())
            print("here")
            for i in range(0,label_size):
                emb = torch.tensor(input_x['emb'][i]).cuda()
                proto = emb.mean(0)
                p = proto.unsqueeze(0).repeat(emb.size(0), 1)
                pdist = nn.PairwiseDistance(p=2)  #
                distance = 1/pdist(emb, p)
                # a, idx1 = torch.sort(distance, descending=True)
                if hold_num!=0:
                    if if_topk == False:
                        idx = torch.multinomial(distance, hold_num, replacement=False)
                        print(idx)
                    else:
                        a, idx1 = torch.sort(distance, descending=True)  #
                        idx = idx1[:hold_num]
                        print(idx)
                    for k in idx:
                        input_x_hold['input_ids'][i].append(input_x['input_ids'][i][k])
                        input_x_hold['attention_mask'][i].append(input_x['attention_mask'][i][k])
                        if type_id == True:
                            input_x_hold['token_type_ids'][i].append(input_x['token_type_ids'][i][k])
                        input_x_hold['text'][i].append(input_x['text'][i][k])
                        input_x_hold['text_label'][i].append(input_x['text_label'][i][k])
                # print("here")
        sample_list = []

        label_list = []
        for i in range(0,label_num):
            for j in range(0,hold_num):
                sample_dict = {}
                sample_dict['text'] = input_x_hold['text'][i][j]
                sample_dict['label'] = input_x_hold['text_label'][i][j]
                label_list.append(i)
                sample_list.append(sample_dict)

        label_list = torch.tensor(label_list).cuda()
        tasks = []
        sample_labels = []
        for sample in sample_list:
            if sample['label'] not in sample_labels:
                sample_labels.append(sample['label'])
                task_zs = {}
                tasks.append(task_zs)
                task_zs['task'] = sample['label']
                task_zs['examples'] = []
                task_zs['examples'].append(sample['text'])
            else:
                task_zs['examples'].append(sample['text'])
        print("here")
        all_entailment_examples = []
        all_non_entailment_examples = []
        nli_train_examples = []
        nli_dev_examples = []
        ENTAILMENT = 'entailment'
        NON_ENTAILMENT = 'non_entailment'
        get_examples = []
        seen_tasks = tasks
        for task_1 in range(len(seen_tasks)):
            examples_1 = seen_tasks[task_1]['examples']
            for j in range(len(examples_1)):
                for task_2 in range(len(seen_tasks)):
                    examples_2 = seen_tasks[task_2]['examples']
                    for k in range(len(examples_2)):
                        if task_1 == task_2:
                            get_examples.append(
                                InputExample(examples_2[k], examples_1[j], ENTAILMENT))
                        else:
                            get_examples.append(InputExample(examples_2[k], examples_1[j], NON_ENTAILMENT))
        ft_B_Classifier = True
        if ft_B_Classifier == True:
            dev_get_examples = random.sample(get_examples, 100)
            model_part2.train(get_examples, dev_get_examples)
        test_label = []
        test_text = []
        for index, batch_data in enumerate(test_data):
            if index == 0:
                test_label = torch.tensor(batch_data[1])
            else:
                test_label = torch.cat((test_label.clone().detach(), torch.tensor(batch_data[1]).clone().detach()))
            for i in batch_data[2]:
                test_text.append(i)
        nli_input = []
        for t in test_text:
            for task_2 in range(len(tasks)):
                examples_2 = tasks[task_2]['examples']
                for k in range(len(examples_2)):
                    nli_input.append(InputExample(examples_2[k],t))
        dim_hold = hold_num
        label_num_test = label_num #10 #10 #2 #10
        v = model_part2.get_output(nli_input, dim_hold, label_num_test)
        print((v.argmax(1)//dim_hold == test_label.squeeze(-1).cuda()).sum())
        pred = v.argmax(1) // dim_hold
        from sklearn.metrics import precision_recall_fscore_support
        p = precision_recall_fscore_support(test_label.squeeze(-1).cpu(), pred.cpu())
        plf = parse_sklearn_log(p)
        print(plf)
        p1.append(plf)
        test_size = len(test_data.dataset.data) #3914 #1500 #3914 #1500 #3914
        m = v.reshape(test_size , -1, dim_hold) #3914
        k = m.sum(2)
        l = k.argmax(1)
        print((l == test_label.squeeze(-1).cuda()).sum())
        p = precision_recall_fscore_support(test_label.squeeze(-1).cpu(), l.cpu())
        plf2 = parse_sklearn_log(p)
        print(plf2)
        p2.append(plf2)
        print("here")
        model_part3 = Net(hold_num * label_num, label_num)
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_part3.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model_part3.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer_part3 = AdamW(optimizer_grouped_parameters, lr=1e-4, eps=1e-8)
        # optimizer_part3 = AdamW(optimizer_grouped_parameters, lr=1e-2, eps=1e-8)
        # optimizer_part3 = AdamW(model_part3.parameters(), lr=1e-2, eps=1e-8)
        crossentropyloss = nn.CrossEntropyLoss()
        epoch_part3 = 400
        outp = model_part2.get_output(get_examples, dim_hold, label_num)
        for epoch2 in range(0, epoch_part3):
            model_part3.train()
            train_size = hold_num * label_num #300 #60 #300
            idx = [i for i in range(0, train_size)]
            random.shuffle(idx)
            input_x = outp[idx]
            label_list = torch.tensor(label_list).cuda()
            input_labels = label_list[idx]
            outputx = model_part3(input_x)
            loss2 = crossentropyloss(outputx, input_labels)
            print(loss2)
            loss2.backward()
            optimizer_part3.step()
            # scheduler_part2.step()
            model_part3.zero_grad()
        pred = model_part3(v).argmax(1)
        p = precision_recall_fscore_support(test_label.squeeze(-1).cpu(), pred.cpu())
        plf3 = parse_sklearn_log(p)
        print(plf3)
        p3.append(plf3)
        print("finetune")
        print(plf_finetune)
    print("FINETUNE")
    print(p4)
    print("P1")
    print(p1)
    print("P2")
    print(p2)
    print("P3")
    print(p3)

if __name__ == "__main__":
    # main(Config)

    # # for model in ['cnn','gated_cnn','stack_gated_cnn']:
    # for model in ['bert_lstm']:
    #  for model in ['cnn']:
    # for model in ['lstm']:
    for model in ['bert']:
        Config['model_type'] = model
        main(Config)
