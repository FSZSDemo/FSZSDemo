
import os
import torch
import copy
import json
import numpy as np
from tqdm import tqdm
from parser_util import get_parser
import logging
from losses import Loss_fn
from encoder import MyModel
from model import TorchModel
from transformers import AdamW, get_linear_schedule_with_warmup

from data_loader import MyDataset, KShotTaskSampler
from collections import defaultdict
from tensorboardX import SummaryWriter

def init_dataloader(args, mode):
    filePath = os.path.join(args.dataFile, mode + '.json')
    if mode == 'train' or mode == 'valid':
        episode_per_epoch = args.episodeTrain
    else:
        episode_per_epoch = args.episodeTest
    dataset = MyDataset(filePath)
    sampler = KShotTaskSampler(dataset, episodes_per_epoch=episode_per_epoch, n=args.numKShot, k=args.numNWay, q=args.numQShot, num_tasks=1)

    return sampler, dataset


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def init_model(args):
    device = torch.device('cuda', args.numDevice)
    torch.cuda.set_device(device)
    model = MyModel(args).to(device)
    return model

def init_optim(args, model):

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    
    return optimizer

def init_lr_scheduler(args, optim):
    '''
    Initialize the learning rate scheduler
    '''
    
    t_total = args.epochs * args.episodeTrain
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    return scheduler

def deal_data(support_set, query_set, episode_labels):

    text, labels = [], []
    for x in support_set:
        text.append(x["text"])
        labels.append(x["label"])
    for x in query_set:
        text.append(x["text"])
        labels.append(x["label"])  
    label_ids = []
    for label in labels:
        tmp = []
        for l in episode_labels:
            if l == label:
                tmp.append(1)
            else:
                tmp.append(0)
        label_ids.append(tmp)

    return text, label_ids

def train(args, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):

    train_writer = SummaryWriter(os.path.join(args.fileModelSave, 'train_log'))
    
    if val_dataloader is None:
        acc_best_state = None
    
    train_loss, epoch_train_loss = [], []
    train_acc, epoch_train_acc = [], []
    train_p, epoch_train_p = [], []
    train_r, epoch_train_r = [], []
    train_f1, epoch_train_f1 = [], []
    train_auc, epoch_train_auc = [], []
    train_topkacc, epoch_train_topkacc = [], []
    val_loss, epoch_val_loss = [], []
    val_acc, epoch_val_acc = [], []
    val_p, epoch_val_p = [], []
    val_r, epoch_val_r = [], []
    val_f1, epoch_val_f1 = [], []
    val_auc, epoch_val_auc = [], []
    val_topkacc, epoch_val_topkacc = [], []
    best_p = 0
    best_r = 0
    best_f1 = 0
    best_acc = 0
    best_auc = 0
    loss_fn = Loss_fn(args)
    
    acc_best_model_path = os.path.join(args.fileModelSave, 'acc_best_model.pth')

    for epoch in range(args.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        model.train()
        
        for  i, batch in tqdm(enumerate(tr_dataloader)):
            optim.zero_grad()
            support_set, query_set, episode_labels = batch
            text, labels = deal_data(support_set, query_set, episode_labels)
           
            model_outputs = model(text)
           
            loss, p, r, f, acc, auc, topk_acc= loss_fn(model_outputs, labels)
            
            loss.backward()
            optim.step()
            lr_scheduler.step()
            train_loss.append(loss.item())
            train_p.append(p)
            train_r.append(r)
            train_f1.append(f)
            train_acc.append(acc)
            train_auc.append(auc)
            train_topkacc.append(topk_acc)
            print('Train Loss: {}, Train p: {}, Train r: {}, Train f1: {},  Train acc: {},  Train auc: {}, Train topk acc: {}'.format(loss, p, r, f, acc, auc, topk_acc))

        avg_loss = np.mean(train_loss[-args.episodeTrain:])
        avg_acc = np.mean(train_acc[-args.episodeTrain:])
        avg_p = np.mean(train_p[-args.episodeTrain:])
        avg_r = np.mean(train_r[-args.episodeTrain:])
        avg_f1 = np.mean(train_f1[-args.episodeTrain:])
        avg_auc = np.mean(train_auc[-args.episodeTrain:])
        avg_topkacc = np.mean(train_topkacc[-args.episodeTrain:])
        print('Avg Train Loss: {}, Avg Train p: {}, Avg Train r: {}, Avg Train f1: {}, Avg Train acc: {}, Avg Train auc: {}, Avg Train topk acc: {}'.format(avg_loss, avg_p, avg_r, avg_f1, avg_acc, avg_auc, avg_topkacc))
        epoch_train_loss.append(avg_loss)
        epoch_train_acc.append(avg_acc)
        epoch_train_p.append(avg_p)
        epoch_train_r.append(avg_r)
        epoch_train_f1.append(avg_f1)
        epoch_train_auc.append(avg_auc)
        epoch_train_topkacc.append(avg_topkacc)

        if val_dataloader is None:
            continue
        with torch.no_grad():
            model.eval()
            
            for batch in tqdm(val_dataloader):
                support_set, query_set, episode_labels = batch
                text, labels = deal_data(support_set, query_set, episode_labels)
                model_outputs = model(text)
                loss, p, r, f, acc, auc, topkacc = loss_fn(model_outputs, labels)
                
                val_loss.append(loss.item())
                val_acc.append(acc)
                val_p.append(p)
                val_r.append(r)
                val_f1.append(f)
                val_auc.append(auc)
                val_topkacc.append(topkacc)
                
            avg_loss = np.mean(val_loss[-args.episodeTrain:])
            avg_acc = np.mean(val_acc[-args.episodeTrain:])
            avg_p = np.mean(val_p[-args.episodeTrain:])
            avg_r = np.mean(val_r[-args.episodeTrain:])
            avg_f1 = np.mean(val_f1[-args.episodeTrain:])
            avg_auc = np.mean(val_auc[-args.episodeTrain:])
            avg_topkacc = np.mean(val_topkacc[-args.episodeTrain:])
            epoch_val_loss.append(avg_loss)
            epoch_val_acc.append(avg_acc)
            epoch_val_p.append(avg_p)
            epoch_val_r.append(avg_r)
            epoch_val_f1.append(avg_f1)
            epoch_val_auc.append(avg_auc)
            epoch_val_topkacc.append(avg_topkacc)

        postfix = ' (Best)' if avg_p >= best_p else ' (Best: {})'.format(
            best_p)
        r_prefix = ' (Best)' if avg_r >= best_r else ' (Best: {})'.format(
            best_r)
        f1_prefix = ' (Best)' if avg_f1 >= best_f1 else ' (Best: {})'.format(
            best_f1)
        acc_prefix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        auc_prefix = ' (Best)' if avg_auc >= best_auc else ' (Best: {})'.format(
            best_auc)
        print('Avg Val Loss: {}, Avg Val p: {}{}, Avg Val r: {}{}, Avg Val f1: {}{}, Avg Val acc: {}{}, Avg Val auc: {}{},  Avg Val topkacc: {}'.format(
            avg_loss, avg_p, postfix, avg_r, r_prefix, avg_f1, f1_prefix, avg_acc, acc_prefix, avg_auc, auc_prefix, avg_topkacc))
   
       

        if avg_acc >= best_acc:
            torch.save(model.state_dict(), acc_best_model_path)
            best_acc = avg_acc
            acc_best_state = model.state_dict()
        
       
    
    for i, t_f in enumerate(train_f1):
        train_writer.add_scalar("Train/F1", t_f, i)
        train_writer.add_scalar("Train/Loss", train_loss[i], i)

    for i, t_f in enumerate(val_f1):
        train_writer.add_scalar("Val/F1", t_f, i)
        train_writer.add_scalar("Val/Loss", val_loss[i], i)

    for name in ['epoch_train_loss', 'epoch_train_p', 'epoch_train_r', 'epoch_train_f1', 'epoch_train_acc', 'epoch_train_auc', 'epoch_train_topkacc', 'epoch_val_loss', 'epoch_val_p', 'epoch_val_r', 'epoch_val_f1', 'epoch_val_acc', 'epoch_val_auc', 'epoch_val_topkacc']:
        save_list_to_file(os.path.join(args.fileModelSave,
                                       name + '.txt'), locals()[name])

    return model
        

def test(args, test_dataloader, model):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    val_p = []
    val_r = []
    val_loss = []
    val_f1 = []
    val_acc = []
    val_auc = []
    val_topkacc = []
    loss_fn = Loss_fn(args)
    with torch.no_grad():
        model.eval()
        
        for batch in tqdm(test_dataloader):
            support_set, query_set, episode_labels = batch
            text, labels = deal_data(support_set, query_set, episode_labels)
            model_outputs = model(text)
            loss, p, r, f, acc, auc, topkacc = loss_fn(model_outputs, labels)
            
            val_loss.append(loss.item())
            val_acc.append(acc)
            val_p.append(p)
            val_r.append(r)
            val_f1.append(f)
            val_auc.append(auc)
            val_topkacc.append(topkacc)
                
        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        avg_p = np.mean(val_p)
        avg_r = np.mean(val_r)
        avg_f1 = np.mean(val_f1)
        avg_auc = np.mean(val_auc)
        avg_topkacc = np.mean(val_topkacc)


        print('Test p: {}'.format(avg_p))
        print('Test r: {}'.format(avg_r))
        print('Test f1: {}'.format(avg_f1))
        print('Test acc: {}'.format(avg_acc))
        print('Test auc: {}'.format(avg_auc))
        print('Test topkacc: {}'.format(avg_topkacc))
        print('Test Loss: {}'.format(avg_loss))

        path = args.fileModelSave + "/test_score.json"
        with open(path, "a+") as fout:
            tmp = {"p": avg_p, "r": avg_r, "f1": avg_f1, "acc": avg_acc, "auc": avg_auc, "Loss": avg_loss}
            fout.write("%s\n" % json.dumps(tmp, ensure_ascii=False))



def write_args_to_josn(args):
    path = args.fileModelSave + "/config.json"
    args = vars(args)
    json_str = json.dumps(args, indent=4)
    with open(path, 'w') as json_file:
        json_file.write(json_str)

from config import Config
from loader import load_data
import torch.nn as nn
import json
from B_Classifier import B_Classifier
import random
import torch.nn.functional as F
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


def main(config):
    if_topk = False
    seed = 0 #5
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    getarg = True
    args = get_parser().parse_args()
    config["label_data_path"] = args.label_dataFile
    config["g_file_name"] = args.generated_dataFile
    BERT_NLI_PATH =  None 
    SAVENAME = 'mymodel.bin'
    model_part2 = B_Classifier(path=BERT_NLI_PATH, args=args)
    if getarg == True:
        if not os.path.isdir(config["model_path"]):
            os.mkdir(config["model_path"])

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
        seen_data = load_data(config["seen_data_path"], config, 'seen') #Never use
        # seen_data = load_data(config["seen_data_path"], config, 'seen')
        # label_data = load_data(config["label_data_path"], config)
        # test_data = load_data(config["label_data_path"], config)
    args = get_parser().parse_args()
    # config["g_file_name"] = args.generated_dataFile
    # config["label_data_path"] = args.labeled_path.
    # train_data = load_data(config["train_data_path"], config)
    
    if not os.path.exists(args.fileModelSave):
        os.makedirs(args.fileModelSave)

    write_args_to_josn(args)

    # model = init_model(args)
    # print(model)

    tr_dataloader, tr_data = init_dataloader(args, 'train')
    # val_dataloader, _ = init_dataloader(args, 'valid')
    test_dataloader, _ = init_dataloader(args, 'test')

    # for batch in tqdm(test_dataloader):
    #     print("here")
    #     for k in tqdm(train_data):
    #         print("here")
    model = TorchModel(config)
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        model = model.cuda()
    hold_num = args.hold_num
    label_num = args.label_num #16 #9 #11 #11 #11 #9 #9 #11 #9 #11 #9 #16 #16 #7 #18 #27 #18 #27 #16 #18 #27
    criterion = nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    weight_decay = 1e-2
    learning_rate = 2e-5
    type_id = False
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # optimizer = AdamW(model.parameters(), lr=learning_rate)
    optimizer1 = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # 训练
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    for epoch in range(0,5):
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
        # for index, batch_data in  enumerate(label_data):
        #     input_ids, labels, text, text_label = batch_data
        #     # index += 1
        #     if type_id == True:
        #         samples = {'input_ids': input_ids['input_ids'].squeeze(1),
        #                    'attention_mask': input_ids['attention_mask'].squeeze(1),
        #                    'token_type_ids': input_ids['token_type_ids'].squeeze(1)}
        #     else:
        #         samples = {'input_ids': input_ids['input_ids'].squeeze(1),
        #                    'attention_mask': input_ids['attention_mask'].squeeze(1)}
        #     for sample in samples:
        #         samples[sample] = samples[sample].cuda()
        #     label_output = model(samples,labels)
        #     # criterion(label_data,batch_data[1])
        #     # loss = criterion(label_output[0],batch_data[1].squeeze(1).cuda())
        #     # loss.backward()
        #     # print(loss)
        #     # optimizer1.step()
        #     # label_output = torch.zeros(len(labels),768).cuda()
        #     # ed_temp = []
        #     # ed_all = []
        #     # for i in range(0,label_data.size()[0]):
        #     #     label_output[labels[i]] = label_data[i]
        #     #     ed_temp.append(0)
        input_x_hold = {}
        input_x_hold['input_ids'] = {}
        input_x_hold['attention_mask'] = {}
        input_x_hold['token_type_ids'] = {}
        input_x_hold['text'] = {}
        input_x_hold['text_label'] = {}
        input_x_hold['score'] = {}
        label_size = label_num
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
        label_num_test = 5

        if_classification = False
        if if_classification==True:
            for i in range(0,2):
                # index = 0
                for index, batch_data in enumerate(train_data):
                    # index += 1
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
                    #     x = model(samples, labels)
                    x = model(samples, labels)
                    if_classification = False
                    loss = criterion(x[0], batch_data[1].squeeze(1).cuda())
                    y_pred_label = x[0].argmax(dim=1)
                    acc = ((y_pred_label == labels.view(-1).cuda()).sum()).item()
                    # loss = criterion(y_pred_prob.view(-1, 2), label.view(-1))
                    loss.backward()
                    l = loss.item()
                    print( str(l) +"----"+str(acc))
                    optimizer1.step()
                label_size = label_output[0].size(0)
            # index = 0
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
                with torch.no_grad():
                    x = model(samples, labels)
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
                pdist = nn.PairwiseDistance(p=2)
                distance = 1/pdist(emb, p)
                if hold_num!=0:
                    if if_topk == False:
                        idx = torch.multinomial(distance, hold_num, replacement=False)
                        print(idx)
                    else:
                        a, idx1 = torch.sort(distance, descending=True)  #e
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

        print("here")
        sample_list = []
        label_list = []
        use_seen = False
        flag = 1
        if use_seen == True and BERT_NLI_PATH == None and flag == 0:
            seen_label_num = 18
            input_x_seen = {}
            input_x_seen['input_ids'] = {}
            input_x_seen['attention_mask'] = {}
            input_x_seen['token_type_ids'] = {}
            input_x_seen['text'] = {}
            input_x_seen['text_label'] = {}
            input_x_seen['score'] = {}
            input_x_seen['emb'] = {}
            input_x_hold_seen = {}
            input_x_hold_seen['input_ids'] = {}
            input_x_hold_seen['attention_mask'] = {}
            input_x_hold_seen['token_type_ids'] = {}
            input_x_hold_seen['text'] = {}
            input_x_hold_seen['text_label'] = {}
            input_x_hold_seen['score'] = {}
            label_size = label_num
            # optimizer1, scheduler1 = get_optimizer(model, 1e-3)
            for label in range(0, seen_label_num):
                output_x[label] = []
                # input_x[label] = []
                input_x_seen['input_ids'][label] = []
                input_x_seen['attention_mask'][label] = []
                input_x_seen['token_type_ids'][label] = []
                input_x_seen['text'][label] = []
                input_x_seen['text_label'][label] = []
                input_x_seen['score'][label] = []
                input_x_seen['emb'][label] = []
                input_x_hold_seen['input_ids'][label] = []
                input_x_hold_seen['attention_mask'][label] = []
                input_x_hold_seen['token_type_ids'][label] = []
                input_x_hold_seen['text'][label] = []
                input_x_hold_seen['text_label'][label] = []
                input_x_hold_seen['score'][label] = []
            for index, batch_data in enumerate(seen_data):
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
                #     x = model(samples, labels)
                with torch.no_grad():
                    x = model(samples, labels)
                    # print("here")
                    sample_num = x.size(0)
                    for i in range(0, sample_num):
                        input_x_seen['input_ids'][labels[i].item()].append(samples['input_ids'][i])
                        input_x_seen['attention_mask'][labels[i].item()].append(samples['attention_mask'][i])
                        if type_id == True:
                            input_x_seen['token_type_ids'][labels[i].item()].append(samples['token_type_ids'][i])
                        input_x_seen['text'][labels[i].item()].append(text[i])
                        input_x_seen['text_label'][labels[i].item()].append(label_text[i])
                        input_x_seen['emb'][labels[i].item()].append(x[i].tolist())
            print("here")
            for i in range(0, seen_label_num):
                emb = torch.tensor(input_x_seen['emb'][i]).cuda()
                proto = emb.mean(0)
                p = proto.unsqueeze(0).repeat(emb.size(0), 1)
                pdist = nn.PairwiseDistance(p=2)  #
                distance = 1 / pdist(emb, p)
                # if if_topk == True:
                # else:
                topk = if_topk
                if hold_num < distance.size(0):
                    if topk == True:
                        a, idx1 = torch.sort(distance, descending=True)  #
                        idx = idx1[:hold_num]
                    else:
                        idx = torch.multinomial(distance, hold_num, replacement=False)
                else:
                    idx = torch.multinomial(distance, distance.size(0), replacement=False)
                for k in idx:
                    input_x_hold_seen['input_ids'][i].append(input_x_seen['input_ids'][i][k])
                    input_x_hold_seen['attention_mask'][i].append(input_x_seen['attention_mask'][i][k])
                    if type_id == True:
                        input_x_hold_seen['token_type_ids'][i].append(input_x_seen['token_type_ids'][i][k])
                    input_x_hold_seen['text'][i].append(input_x_seen['text'][i][k])
                    input_x_hold_seen['text_label'][i].append(input_x_seen['text_label'][i][k])
                # print("here")
            seen_sample_list = []
            seen_label_list = []
            for i in range(0, seen_label_num):
                for j in range(0, len(input_x_hold_seen['text'][i])):
                    sample_dict = {}
                    sample_dict['text'] = input_x_hold_seen['text'][i][j]
                    sample_dict['label'] = input_x_hold_seen['text_label'][i][j]
                    seen_label_list.append(i)
                    seen_sample_list.append(sample_dict)

            seen_label_list = torch.tensor(seen_label_list).cuda()
            tasks = []
            sample_labels = []
            for sample in seen_sample_list:
                if sample['label'] not in sample_labels:
                    sample_labels.append(sample['label'])
                    task_z = {}
                    tasks.append(task_z)
                    task_z['task'] = sample['label']
                    task_z['examples'] = []
                    task_z['examples'].append(sample['text'])
                else:
                    task_z['examples'].append(sample['text'])
            # with open("./CLINC-TOP20.json", "w") as f:
            # for i in sample_list:
            #     line = json.dumps(i, ensure_ascii=False)
            #     f.write(line + '\n')
            print("here")
            all_entailment_examples = []
            all_non_entailment_examples = []
            nli_train_examples = []
            nli_dev_examples = []
            ENTAILMENT = 'entailment'
            NON_ENTAILMENT = 'non_entailment'
            pretrain_seen = True
            if pretrain_seen == True:
                for task in tasks:
                    examples = task['examples']
                    for j in range(len(examples)):
                        for k in range(len(examples)):
                            if k <= j:
                                continue

                            all_entailment_examples.append(InputExample(examples[j], examples[k], ENTAILMENT))
                            all_entailment_examples.append(InputExample(examples[k], examples[j], ENTAILMENT))
                for task_1 in range(len(tasks)):
                    for task_2 in range(len(tasks)):
                        if task_2 <= task_1:
                            continue
                        examples_1 = tasks[task_1]['examples']
                        examples_2 = tasks[task_2]['examples']
                        for j in range(len(examples_1)):
                            for k in range(len(examples_2)):
                                all_non_entailment_examples.append(
                                    InputExample(examples_1[j], examples_2[k], NON_ENTAILMENT))
                                all_non_entailment_examples.append(
                                    InputExample(examples_2[k], examples_1[j], NON_ENTAILMENT))

                nli_train_examples.append(all_entailment_examples + all_non_entailment_examples)
                nli_dev_examples.append(
                    all_entailment_examples[:100] + all_non_entailment_examples[:100])  # sanity check for over-fitting
                model_part2.train(nli_train_examples[0], nli_dev_examples[0], "not-earlystop")

            #
            #     seen_task_list = []
            #     list = random.sample(data_task[i], hold_num)
            #     seen_task['examples'] = list
            #     seen_tasks.append(seen_task)
            #
            # seen_label_num = len(seen_tasks)
            # for i in range(0, seen_label_num):
            #     for j in range(0, hold_num):
            #         sample_dict = {}
            #         sample_dict['text'] = input_x_hold['text'][i][j]
            #         sample_dict['label'] = input_x_hold['text_label'][i][j]
            #         label_list.append(i)
            #         sample_list.append(sample_dict)
            torch.save(model_part2.model.state_dict(), SAVENAME)
            # torch.save(model.state_dict(), SAVENAME)
            # torch.save(model.state_dict(), acc_best_model_path)

        if use_seen == True and BERT_NLI_PATH == None and flag == 1:
            seen_label_num = 18
            input_x_seen = {}
            input_x_seen['input_ids'] = {}
            input_x_seen['attention_mask'] = {}
            input_x_seen['token_type_ids'] = {}
            input_x_seen['text'] = {}
            input_x_seen['text_label'] = {}
            input_x_seen['score'] = {}
            input_x_seen['emb'] = {}
            input_x_hold_seen = {}
            input_x_hold_seen['input_ids'] = {}
            input_x_hold_seen['attention_mask'] = {}
            input_x_hold_seen['token_type_ids'] = {}
            input_x_hold_seen['text'] = {}
            input_x_hold_seen['text_label'] = {}
            input_x_hold_seen['score'] = {}
            label_size = label_num
            # optimizer1, scheduler1 = get_optimizer(model, 1e-3)
            for label in range(0, seen_label_num):
                output_x[label] = []
                # input_x[label] = []
                input_x_seen['input_ids'][label] = []
                input_x_seen['attention_mask'][label] = []
                input_x_seen['token_type_ids'][label] = []
                input_x_seen['text'][label] = []
                input_x_seen['text_label'][label] = []
                input_x_seen['score'][label] = []
                input_x_seen['emb'][label] = []
                input_x_hold_seen['input_ids'][label] = []
                input_x_hold_seen['attention_mask'][label] = []
                input_x_hold_seen['token_type_ids'][label] = []
                input_x_hold_seen['text'][label] = []
                input_x_hold_seen['text_label'][label] = []
                input_x_hold_seen['score'][label] = []
            for index, batch_data in enumerate(seen_data):
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
                # 输入变化时这里需要修改，比如多输入，多输出的情况
                # with torch.no_grad():
                #     x = model(samples, labels)
                with torch.no_grad():
                    x = model(samples, labels)
                    # print("here")
                    sample_num = x.size(0)
                    for i in range(0, sample_num):
                        input_x_seen['input_ids'][labels[i].item()].append(samples['input_ids'][i])
                        input_x_seen['attention_mask'][labels[i].item()].append(samples['attention_mask'][i])
                        if type_id == True:
                            input_x_seen['token_type_ids'][labels[i].item()].append(samples['token_type_ids'][i])
                        input_x_seen['text'][labels[i].item()].append(text[i])
                        input_x_seen['text_label'][labels[i].item()].append(label_text[i])
                        input_x_seen['emb'][labels[i].item()].append(x[i].tolist())
            print("here")
            for i in range(0, seen_label_num):
                emb = torch.tensor(input_x_seen['emb'][i]).cuda()
                proto = emb.mean(0)
                p = proto.unsqueeze(0).repeat(emb.size(0), 1)
                pdist = nn.PairwiseDistance(p=2)
                distance = 1 / pdist(emb, p)
                if if_topk == True:
                    a, idx1 = torch.sort(distance, descending=True)  #
                    idx = idx1[:hold_num]
                else:
                    idx = torch.multinomial(distance, distance.size(0), replacement=False)
                for k in idx:
                    input_x_hold_seen['input_ids'][i].append(input_x_seen['input_ids'][i][k])
                    input_x_hold_seen['attention_mask'][i].append(input_x_seen['attention_mask'][i][k])
                    if type_id == True:
                        input_x_hold_seen['token_type_ids'][i].append(input_x_seen['token_type_ids'][i][k])
                    input_x_hold_seen['text'][i].append(input_x_seen['text'][i][k])
                    input_x_hold_seen['text_label'][i].append(input_x_seen['text_label'][i][k])
                # print("here")
            seen_sample_list = []
            seen_label_list = []
            for i in range(0, seen_label_num):
                for j in range(0, len(input_x_hold_seen['text'][i])):
                    sample_dict = {}
                    sample_dict['text'] = input_x_hold_seen['text'][i][j]
                    sample_dict['label'] = input_x_hold_seen['text_label'][i][j]
                    seen_label_list.append(i)
                    seen_sample_list.append(sample_dict)

            seen_label_list = torch.tensor(seen_label_list).cuda()
            tasks = []
            sample_labels = []
            for sample in seen_sample_list:
                if sample['label'] not in sample_labels:
                    sample_labels.append(sample['label'])
                    task_z = {}
                    tasks.append(task_z)
                    task_z['task'] = sample['label']
                    task_z['examples'] = []
                    task_z['examples'].append(sample['text'])
                else:
                    task_z['examples'].append(sample['text'])
            # with open("./CLINC-TOP20.json", "w") as f:
            # for i in sample_list:
            #     line = json.dumps(i, ensure_ascii=False)
            #     f.write(line + '\n')
            print("here")
            all_entailment_examples = []
            all_non_entailment_examples = []
            nli_train_examples = []
            nli_dev_examples = []
            ENTAILMENT = 'entailment'
            NON_ENTAILMENT = 'non_entailment'
            pretrain_seen = True
            if pretrain_seen == True:
                for task in tasks:
                    train_examples = []
                    examples = task['examples']
                    other_examples = []
                    for t in tasks:
                        if t == task:
                            continue
                        other_examples.extend(t['examples'])
                    example_len = len(examples)
                    if example_len > 50:
                        l = [i for i in range(0, example_len)]
                        example_len = 50
                        examples_idx = random.sample(l, example_len)
                    else:
                        examples_idx = [i for i in range(0, example_len)]
                    for j in examples_idx:
                        l = [i for i in range(0, len(examples))]
                        examples_idx2 = random.sample(l, example_len)
                        for k in examples_idx2:
                            # if k <= j:
                            #     continue
                            train_examples.append(InputExample(examples[j], examples[k], ENTAILMENT))
                            # all_entailment_examples.append(InputExample(examples[k], examples[j], ENTAILMENT))
                        # example_len = len(examples)
                        non_examples = random.sample(other_examples, example_len)
                        for t in non_examples:
                            train_examples.append(
                                InputExample(examples[j], t, NON_ENTAILMENT))
                    random.shuffle(train_examples)
                    nli_train_examples.extend(train_examples)

                # for task_1 in range(len(tasks)):
                #     for task_2 in range(len(tasks)):
                #         if task_2 <= task_1:
                #             continue
                #         examples_1 = tasks[task_1]['examples']
                #         examples_2 = tasks[task_2]['examples']
                #         for j in range(len(examples_1)):
                #             for k in range(len(examples_2)):
                #                 all_non_entailment_examples.append(
                #                     InputExample(examples_1[j], examples_2[k], NON_ENTAILMENT))
                #                 all_non_entailment_examples.append(
                #                     InputExample(examples_2[k], examples_1[j], NON_ENTAILMENT))
                print("预训练数据准备好了")
                nli_dev_examples = random.sample(nli_train_examples, 200)
                # nli_train_examples.append(all_entailment_examples + all_non_entailment_examples)
                # nli_dev_examples.append(
                #     all_entailment_examples[:100] + all_non_entailment_examples[:100])  # sanity check for over-fitting

                model_part2.train(nli_train_examples, nli_dev_examples, "not-earlystop")
            # data = tr_data.df
            # seen_label_list = []
            # data_task = {}
            # for k in range(0, len(data['text'])):
            #     if data['class_name'][k] not in seen_label_list:
            #         seen_label_list.append(data['class_name'][k])
            #         data_task[data['class_name'][k]] = []
            #     data_task[data['class_name'][k]].append(data['text'][k])
            #
            # seen_tasks = []
            # for i in data_task:
            #     seen_task = {}
            #     seen_task['task'] = i
            #     x = model(samples, labels)
            #     # print("here")
            #     sample_num = x.size(0)
            #     for i in range(0, sample_num):
            #         input_x['input_ids'][labels[i].item()].append(samples['input_ids'][i])
            #         input_x['attention_mask'][labels[i].item()].append(samples['attention_mask'][i])
            #         if type_id == True:
            #             input_x['token_type_ids'][labels[i].item()].append(samples['token_type_ids'][i])
            #         input_x['text'][labels[i].item()].append(text[i])
            #         input_x['text_label'][labels[i].item()].append(label_text[i])
            #         input_x['emb'][labels[i].item()].append(x[i].tolist())
            #
            #
            #     seen_task_list = []
            #     list = random.sample(data_task[i], hold_num)
            #     seen_task['examples'] = list
            #     seen_tasks.append(seen_task)
            #
            # seen_label_num = len(seen_tasks)
            # for i in range(0, seen_label_num):
            #     for j in range(0, hold_num):
            #         sample_dict = {}
            #         sample_dict['text'] = input_x_hold['text'][i][j]
            #         sample_dict['label'] = input_x_hold['text_label'][i][j]
            #         label_list.append(i)
            #         sample_list.append(sample_dict)
            torch.save(model_part2.model.state_dict(), SAVENAME)
            # torch.save(model.state_dict(), acc_best_model_path)

        if use_seen == True and BERT_NLI_PATH == None and flag == 2:
            seen_label_num = 18
            input_x_seen = {}
            input_x_seen['input_ids'] = {}
            input_x_seen['attention_mask'] = {}
            input_x_seen['token_type_ids'] = {}
            input_x_seen['text'] = {}
            input_x_seen['text_label'] = {}
            input_x_seen['score'] = {}
            input_x_seen['emb'] = {}
            input_x_hold_seen = {}
            input_x_hold_seen['input_ids'] = {}
            input_x_hold_seen['attention_mask'] = {}
            input_x_hold_seen['token_type_ids'] = {}
            input_x_hold_seen['text'] = {}
            input_x_hold_seen['text_label'] = {}
            input_x_hold_seen['score'] = {}
            label_size = label_num
            # optimizer1, scheduler1 = get_optimizer(model, 1e-3)
            for label in range(0, seen_label_num):
                output_x[label] = []
                # input_x[label] = []
                input_x_seen['input_ids'][label] = []
                input_x_seen['attention_mask'][label] = []
                input_x_seen['token_type_ids'][label] = []
                input_x_seen['text'][label] = []
                input_x_seen['text_label'][label] = []
                input_x_seen['score'][label] = []
                input_x_seen['emb'][label] = []
                input_x_hold_seen['input_ids'][label] = []
                input_x_hold_seen['attention_mask'][label] = []
                input_x_hold_seen['token_type_ids'][label] = []
                input_x_hold_seen['text'][label] = []
                input_x_hold_seen['text_label'][label] = []
                input_x_hold_seen['score'][label] = []
            for index, batch_data in enumerate(seen_data):
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
                #     x = model(samples, labels)
                with torch.no_grad():
                    x = model(samples, labels)
                    # print("here")
                    sample_num = x.size(0)
                    for i in range(0, sample_num):
                        input_x_seen['input_ids'][labels[i].item()].append(samples['input_ids'][i])
                        input_x_seen['attention_mask'][labels[i].item()].append(samples['attention_mask'][i])
                        if type_id == True:
                            input_x_seen['token_type_ids'][labels[i].item()].append(samples['token_type_ids'][i])
                        input_x_seen['text'][labels[i].item()].append(text[i])
                        input_x_seen['text_label'][labels[i].item()].append(label_text[i])
                        input_x_seen['emb'][labels[i].item()].append(x[i].tolist())
            print("here")
            for i in range(0, seen_label_num):
                emb = torch.tensor(input_x_seen['emb'][i]).cuda()
                proto = emb.mean(0)
                p = proto.unsqueeze(0).repeat(emb.size(0), 1)
                pdist = nn.PairwiseDistance(p=2)
                distance = 1 / pdist(emb, p)
                if if_topk == True:
                    a, idx1 = torch.sort(distance, descending=True)
                    idx = idx1[:hold_num]
                else:
                    idx = torch.multinomial(distance, distance.size(0), replacement=False)
                for k in idx:
                    input_x_hold_seen['input_ids'][i].append(input_x_seen['input_ids'][i][k])
                    input_x_hold_seen['attention_mask'][i].append(input_x_seen['attention_mask'][i][k])
                    if type_id == True:
                        input_x_hold_seen['token_type_ids'][i].append(input_x_seen['token_type_ids'][i][k])
                    input_x_hold_seen['text'][i].append(input_x_seen['text'][i][k])
                    input_x_hold_seen['text_label'][i].append(input_x_seen['text_label'][i][k])
                # print("here")
            seen_sample_list = []
            seen_label_list = []
            for i in range(0, seen_label_num):
                for j in range(0, len(input_x_hold_seen['text'][i])):
                    sample_dict = {}
                    sample_dict['text'] = input_x_hold_seen['text'][i][j]
                    sample_dict['label'] = input_x_hold_seen['text_label'][i][j]
                    seen_label_list.append(i)
                    seen_sample_list.append(sample_dict)

            seen_label_list = torch.tensor(seen_label_list).cuda()
            tasks = []
            sample_labels = []
            for sample in seen_sample_list:
                if sample['label'] not in sample_labels:
                    sample_labels.append(sample['label'])
                    task_z = {}
                    tasks.append(task_z)
                    task_z['task'] = sample['label']
                    task_z['examples'] = []
                    task_z['examples'].append(sample['text'])
                else:
                    task_z['examples'].append(sample['text'])
            # with open("./CLINC-TOP20.json", "w") as f:
            # for i in sample_list:
            #     line = json.dumps(i, ensure_ascii=False)
            #     f.write(line + '\n')
            print("here")
            all_entailment_examples = []
            all_non_entailment_examples = []
            nli_train_examples = []
            nli_dev_examples = []
            ENTAILMENT = 'entailment'
            NON_ENTAILMENT = 'non_entailment'
            pretrain_seen = True
            if pretrain_seen == True:
                for task in tasks:
                    train_examples = []
                    examples = task['examples']
                    other_examples = []
                    for t in tasks:
                        if t == task:
                            continue
                        other_examples.extend(t['examples'])
                    example_len = len(examples)
                    np.random.sample(example_len,40000,replace = True)
                    if example_len > 200:
                        l = [i for i in range(0, example_len)]
                        example_len = 200
                        examples_idx = random.sample(l, example_len)
                    else:
                        examples_idx = [i for i in range(0, example_len)]
                    for j in examples_idx:
                        l = [i for i in range(0, len(examples))]
                        examples_idx2 = random.sample(l, example_len)
                        for k in examples_idx2:
                            # if k <= j:
                            #     continue
                            train_examples.append(InputExample(examples[j], examples[k], ENTAILMENT))
                            # all_entailment_examples.append(InputExample(examples[k], examples[j], ENTAILMENT))
                        # example_len = len(examples)
                        non_examples = random.sample(other_examples, example_len)
                        for t in non_examples:
                            train_examples.append(
                                InputExample(examples[j], t, NON_ENTAILMENT))
                    random.shuffle(train_examples)
                    nli_train_examples.extend(train_examples)

                # for task_1 in range(len(tasks)):
                #     for task_2 in range(len(tasks)):
                #         if task_2 <= task_1:
                #             continue
                #         examples_1 = tasks[task_1]['examples']
                #         examples_2 = tasks[task_2]['examples']
                #         for j in range(len(examples_1)):
                #             for k in range(len(examples_2)):
                #                 all_non_entailment_examples.append(
                #                     InputExample(examples_1[j], examples_2[k], NON_ENTAILMENT))
                #                 all_non_entailment_examples.append(
                #                     InputExample(examples_2[k], examples_1[j], NON_ENTAILMENT))

                nli_dev_examples = random.sample(nli_train_examples, 200)
                # nli_train_examples.append(all_entailment_examples + all_non_entailment_examples)
                # nli_dev_examples.append(
                #     all_entailment_examples[:100] + all_non_entailment_examples[:100])  # sanity check for over-fitting

                model_part2.train(nli_train_examples, nli_dev_examples, "not-earlystop")

            torch.save(model_part2.model.state_dict(), SAVENAME)
            # torch.save(model.state_dict(), acc_best_model_path)
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
                task_z = {}
                tasks.append(task_z)
                task_z['task'] = sample['label']
                task_z['examples'] = []
                task_z['examples'].append(sample['text'])
            else:
                task_z['examples'].append(sample['text'])
        # with open("./CLINC-TOP20.json", "w") as f:
            # for i in sample_list:
            #     line = json.dumps(i, ensure_ascii=False)
            #     f.write(line + '\n')
        print("here")
        all_entailment_examples = []
        all_non_entailment_examples = []
        nli_train_examples = []
        nli_dev_examples = []
        ENTAILMENT = 'entailment'
        NON_ENTAILMENT = 'non_entailment'
        pretrain = False
        if pretrain == True:
            for task in tasks:
                examples = task['examples']
                for j in range(len(examples)):
                    for k in range(len(examples)):
                        if k <= j:
                            continue

                        all_entailment_examples.append(InputExample(examples[j], examples[k], ENTAILMENT))
                        all_entailment_examples.append(InputExample(examples[k], examples[j], ENTAILMENT))
            for task_1 in range(len(tasks)):
                for task_2 in range(len(tasks)):
                    if task_2 <= task_1:
                        continue
                    examples_1 = tasks[task_1]['examples']
                    examples_2 = tasks[task_2]['examples']
                    for j in range(len(examples_1)):
                        for k in range(len(examples_2)):
                            all_non_entailment_examples.append(
                                InputExample(examples_1[j], examples_2[k], NON_ENTAILMENT))
                            all_non_entailment_examples.append(
                                InputExample(examples_2[k], examples_1[j], NON_ENTAILMENT))

            nli_train_examples.append(all_entailment_examples + all_non_entailment_examples)
            nli_dev_examples.append(all_entailment_examples[:100] + all_non_entailment_examples[:100])  # sanity check for over-fitting

            model_part2.train(nli_train_examples[0], nli_dev_examples[0])
        acc_list = []
        acc_list2 = []
        acc_list3 = []
        part3_use_g = False
        if part3_use_g == True:
            get_examples = []
            for task_1 in range(len(tasks)):
                examples_1 = tasks[task_1]['examples']
                for j in range(len(examples_1)):
                    for task_2 in range(len(tasks)):
                        examples_2 = tasks[task_2]['examples']
                        for k in range(len(examples_2)):
                            if task_1 == task_2:
                                get_examples.append(
                                    InputExample(examples_1[j], examples_2[k], ENTAILMENT))
                            else:
                                get_examples.append(InputExample(examples_1[j], examples_2[k], NON_ENTAILMENT))
        else:
            test_d = []
            for index, batch_data in enumerate(test_dataloader):
                test_d.append(batch_data)
            test_d2 = []
            # for index, batch_data in enumerate(test_dataloader):
            index = 0
            for batch in test_d:
                print(index)
                index += 1
                B_Classifier_copy = copy.deepcopy(model_part2)
                get_examples = []
                g_tasks = []
                batch_dict = {}
                label_id = 0
                label_list = []
                seen_tasks = []
                for i in range(0, len(batch[2])):
                    batch_dict[batch[2][i]] = []
                if isinstance (batch[0],int) == False:
                    for i in range(0, len(batch[0])):
                        batch_dict[batch[0][i]['label']].append(batch[0][i]['text'])
                for i in batch_dict:
                    seen_task = {}
                    seen_task['task'] = i
                    seen_task['examples'] = batch_dict[i]
                    seen_tasks.append(seen_task)
                    for t in tasks:
                        if t['task'] == i:
                            g_tasks.append(t)
                            seen_task['examples'].extend(t['examples'])
                            continue
                    for t in seen_task['examples']:
                        label_list.append(label_id)
                    label_id += 1
                # print("here")
                get_examples = []
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
                dim_hold = len(seen_tasks[0]['examples'])
                model_part3 = Net(dim_hold * label_num_test, label_num_test)
                ft_B_Classifier = True
                if ft_B_Classifier == True:
                    dev_get_examples = random.sample(get_examples,100)
                    B_Classifier_copy.train(get_examples, dev_get_examples)
                outp = B_Classifier_copy.get_output(get_examples, dim_hold, 5)
                print("here")
                optimizer_grouped_parameters = [
                    {'params': [p for n, p in model_part3.named_parameters() if not any(nd in n for nd in no_decay)],
                     'weight_decay': weight_decay},
                    {'params': [p for n, p in model_part3.named_parameters() if any(nd in n for nd in no_decay)],
                     'weight_decay': 0.0}
                ]
                optimizer_part3 = AdamW(optimizer_grouped_parameters, lr=1e-2, eps=1e-8)
                crossentropyloss = nn.CrossEntropyLoss()
                epoch_part3 = 100
                shot_num = args.numKShot
                sample_num = label_num_test * (shot_num + hold_num)
                for epoch2 in range(0, epoch_part3):
                    model_part3.train()
                    idx = [i for i in range(0, sample_num)]
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
                for batch_test in batch[1]:
                    unseen_tasks = []
                    unseen_batch_dict = {}
                    for i in range(0, len(batch[2])):
                        unseen_batch_dict[batch[2][i]] = []
                    for i in range(0, len(batch_test)):
                        unseen_batch_dict[batch_test[i]['label']].append(batch_test[i]['text'])
                    for i in batch_dict:
                        unseen_task = {}
                        unseen_task['task'] = i
                        unseen_task['examples'] = unseen_batch_dict[i]
                        unseen_tasks.append(unseen_task)
                    test_label = []
                    test_text = []
                    nli_input = []
                    for task_1 in range(len(unseen_tasks)):
                        examples_1 = unseen_tasks[task_1]['examples']
                        for j in range(len(examples_1)):
                            for task_2 in range(len(seen_tasks)):
                                examples_2 = seen_tasks[task_2]['examples']
                                for k in range(len(examples_2)):
                                    nli_input.append(InputExample(examples_2[k], examples_1[j]))
                    print("here")
                    target = []
                    for i in range(0, len(unseen_tasks)):
                        for t in unseen_tasks[i]['examples']:
                            target.append(i)
                    target = torch.tensor(target).cuda()
                    v = B_Classifier_copy.get_output(nli_input, dim_hold, label_num_test)
                    model_part3.eval()
                    pred = model_part3(v).argmax(1)
                    from sklearn.metrics import precision_recall_fscore_support

                    p = precision_recall_fscore_support(target.cpu(), pred.cpu())
                    plf = parse_sklearn_log(p)
                    print(plf)
                    acc_list.append(plf['micro_rec_unseen'])
                    a = torch.tensor(acc_list)
                    a = a.mean()
                    print("REC---p1")
                    print(a)
                    print(BERT_NLI_PATH)
                    num = shot_num + hold_num
                    pred = v.argmax(1)//num

                    p = precision_recall_fscore_support(target.cpu(), pred.cpu())
                    plf2 = parse_sklearn_log(p)
                    acc_list2.append(plf2['micro_rec_unseen'])
                    a = torch.tensor(acc_list2)
                    a = a.mean()
                    print(plf2)
                    print("REC---p2")
                    print(a)

                    m = v.reshape(125, -1, num)
                    k = m.sum(2)
                    l = k.argmax(1)
                    p = precision_recall_fscore_support(target.cpu(), l.cpu())
                    plf3 = parse_sklearn_log(p)
                    print(plf3)
                    acc_list3.append(plf3['micro_rec_unseen'])
                    a = torch.tensor(acc_list3)
                    a = a.mean()
                    print("REC---p3")
                    print(a)
        a = torch.tensor(acc_list)
        a = a.mean()
        print("REC!!!")
        print(a)
        a2 = torch.tensor(acc_list2)
        a2 = a2.mean()
        print("REC!!!")
        print(a2)
        a3 = torch.tensor(acc_list3)
        a3 = a3.mean()
        print("REC!!!")
        print(a3)
        print("here")
        if epoch==1:
            with open(config["g_file_name"]+"rec"+str(args.numKShot)+".txt", "w") as f:
                # for i in output['labels']:
                for i in range(0,len(acc_list)):
                    f.write(str(acc_list[i])+'\n')
                f.write("--------------")
                for i in range(0, len(acc_list2)):
                    f.write(str(acc_list2[i]) + '\n')
                f.write("--------------")
                for i in range(0, len(acc_list3)):
                    f.write(str(acc_list3[i]) + '\n')
                f.write("--------------")
                f.write(str(a) + '\n')
                f.write(str(a2) + '\n')
                f.write(str(a3) + '\n')
if __name__ == '__main__':
    # config = Config
    # path = config["g_file_name"]
    # for i in (0,5):
    #     config["label_data_path"]= path + '0'+ str(i) + 'amazon' + str(i) +".json"
    #     config["g_file_name"]= path + '0'+str(i)+'amazon'+str(i)+"-dataset.jsonl"
    #     config["data_file"]="../data/Amazon/0"+str(i)
    main(Config)
