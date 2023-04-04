'''
Author: xiaoyichao xiao_yi_chao@163.com
Date: 2023-02-28 18:58:21
LastEditors: xiaoyichao xiao_yi_chao@163.com
LastEditTime: 2023-02-28 19:06:14
FilePath: /bert_transformer/bert/also_cool.py
Description: 这是一件很cool的事情

tensorboard --logdir=bert/experiment_log

'''
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import pickle
import common4bert
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split
# from three_piece_tokenizer import ThreePieceTokenizer
from model import BERTClassifier, DistilBERTIntent
# from transformer_model_ import  MyBertModel
from intent_data_loader import IntentDataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer, BertModel, BertConfig
# 注意，这个位置要引入私有包
# pip install -i https://mirrors.haohaozhu.me/artifactory/api/pypi/pypi/simple/  transformers4token --upgrade
# from transformers4token import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from sklearn.metrics import accuracy_score, ndcg_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

# define hyperparameters
max_length = 64
pkl_examples_limit = 200
num_labels = 3
batch_size = 64
epochs = 100
lr = 1e-5

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

writer = SummaryWriter('./experiment')

data_dir_path = "/data/search_opt_model/topk_opt/rank_fine_row_cv_userprofile"


# load tokenizer and model
# my_bert_path = "/data/search_opt_model/topk_opt/distilbert/distilbert_torch"
my_bert_path = "/data/xiaoyichao/projects/bert_transformer/bert/models/bert-base-cased-hhz"
# my_bert_path = "bert-base-chinese"

# 读取数据
# all_pkl_names, all_pkl_paths, _ = common4bert.get_models(data_dir_path, False)
# pkl_path = all_pkl_paths[-1]
with open("data/cleared_data.tsv") as f:
    lines = f.readlines()
    data = []
    label_map = {}
    next_label_id = 0
    for line in lines:
        query, intent = line.strip().split("\t")
        intent = intent.split("--")[0]
        if intent not in label_map:
            label_map[intent] = next_label_id
            next_label_id += 1
            
        if label_map[intent] in [2, 3, 4]:
            data.append([query, label_map[intent]])
num_labels = len(label_map)


tokenizer = BertTokenizer.from_pretrained(my_bert_path) 
config = AutoConfig.from_pretrained(my_bert_path, num_labels=num_labels)

config.num_labels = num_labels


# distilbert = MyBertModel.from_pretrained(my_bert_path, config=config)
distilbert = AutoModel.from_pretrained(my_bert_path, config=config)

# quantization_bit = 4
# distilbert = distilbert.quantize(quantization_bit)
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')



# data = [["客厅", 0] for _ in range(1000)]
# data = [["客厅", 0], ["厨房", 0], ["卫生间", 0], ["冰箱", 0], ["洗衣机", 0], ["电视", 0],["客厅", 0], ["厨房", 0], ["卫生间", 0], ["冰箱", 0], ["洗衣机", 0], ["电视", 0]] 
dataset = IntentDataset(tokenizer=tokenizer, data=data)
encoding = dataset.__getitem__(0)
print("encoding: ", encoding)
print("pkl数据总长度: ", dataset.__len__())


train_size = int(0.8*dataset.__len__())
valid_size = dataset.__len__() - train_size
print("train数据长度: ", train_size)
print("valid数据长度: ", valid_size)
train_dataset, valid_dataset  = random_split(dataset,[train_size, valid_size])

train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader =  DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)



# 创建模型

model = DistilBERTIntent(distilbert, config)

# print_size_of_model(model)
# model = model.half()
# print_size_of_model(model)
# 目前半精度的loss会丢失

# 初始化 GradScaler
scaler = GradScaler()

model = model.to(device)

print("model.state_dict().keys()", list(model.state_dict().keys()))


# define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# define the training loop
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch_idx, batch in enumerate(loader):

        encoder_embedding = batch
        labels = encoder_embedding["label"]

        # optimizer.zero_grad()
        # logits, pred = model(encoder_embedding)
        # loss = criterion(logits.view(-1, config.num_labels), labels)
        # acc = accuracy_score(labels.tolist(), pred.tolist())
        # loss.backward()
        # optimizer.step()
        # epoch_loss += loss.item()
        # epoch_acc += acc

        with autocast():
            logits, pred = model(encoder_embedding)
            loss = criterion(logits.view(-1, config.num_labels), labels)

        optimizer.zero_grad()
        acc = accuracy_score(labels.tolist(), pred.tolist())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        epoch_acc += acc

            
        # ...log the running loss
        writer.add_scalar('training loss', loss.item(), len(loader) * epoch + batch_idx)

        writer.add_scalar('training acc', acc, len(loader) * epoch + batch_idx)

    return epoch_loss / len(loader), epoch_acc / len(loader)

# define the evaluation loop
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            encoder_embedding = batch
            labels = encoder_embedding["label"]
            
            logits, pred = model(encoder_embedding)
            loss = criterion(logits.view(-1, config.num_labels), labels)
            acc = accuracy_score(labels.tolist(), pred.tolist())
            epoch_loss += loss.item()
            epoch_acc += acc
            writer.add_scalar('valid loss', loss.item(), len(loader) * epoch + batch_idx)

            writer.add_scalar('valid acc', acc, len(loader) * epoch + batch_idx)
    return epoch_loss / len(loader), epoch_acc / len(loader)


for epoch in range(epochs):
    train_loss, train_acc = train(model,train_loader,optimizer, criterion, epoch)
    test_loss, test_acc = evaluate(model, valid_loader, criterion)
    
    print(f"Epoch {epoch+1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\tValid Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
