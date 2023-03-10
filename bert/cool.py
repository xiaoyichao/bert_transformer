'''
Author: xiaoyichao xiao_yi_chao@163.com
Date: 2023-02-28 18:58:21
LastEditors: xiaoyichao xiao_yi_chao@163.com
LastEditTime: 2023-02-28 19:06:14
FilePath: /bert_transformer/bert/also_cool.py
Description: 这是一件很cool的事情
'''
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import pickle
import common4bert
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split
from three_piece_tokenizer import ThreePieceTokenizer
from model import BERTClassifier
from search_data_loader import SearchDataset
# 注意，这个位置要引入私有包
# pip install -i https://mirrors.haohaozhu.me/artifactory/api/pypi/pypi/simple/  transformers4token --upgrade
from transformers4token import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer, BertModel, BertConfig, DistilBertTokenizer, DistilBertModel, DistilBertConfig
from sklearn.metrics import accuracy_score, ndcg_score
from sklearn.model_selection import train_test_split

# define hyperparameters
max_length = 64
pkl_examples_limit = 200
num_labels = 4
batch_size = 32
epochs = 2
lr = 1e-5

if torch.cuda.is_available():
    print("torch.cuda.current_device()", torch.cuda.current_device())
else:
    print("USE CPU")

data_dir_path = "/data/search_opt_model/topk_opt/rank_fine_row_cv_userprofile"


# load tokenizer and model
my_bert_path = "/data/search_opt_model/topk_opt/distilbert/distilbert_torch"

tokenizer = ThreePieceTokenizer.from_pretrained(my_bert_path) # 使用自己的三段式tokenizer
config = DistilBertConfig.from_pretrained(my_bert_path, num_labels=num_labels)
distilbert = DistilBertModel.from_pretrained(my_bert_path, config=config, load_in_8bit=True)
# print(list(distilbert.state_dict().keys()))

# 读取数据
all_pkl_names, all_pkl_paths, _ = common4bert.get_models(data_dir_path, False)
pkl_path = all_pkl_paths[-1]
dataset = SearchDataset(pkl_file=pkl_path,max_length=max_length, tokenizer=tokenizer, pkl_examples_limit=pkl_examples_limit)
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
model = BERTClassifier(distilbert, config)
print("model.state_dict().keys()", list(model.state_dict().keys()))


# define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# define the training loop
def train(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch in loader:
        input_ids, attention_mask, token_type_ids, labels = batch
        # input_ids, attention_mask,  labels = batch
        optimizer.zero_grad()
        logits, pred = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(logits.view(-1, config.num_labels), labels)
        acc = accuracy_score(labels.tolist(), pred.tolist())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)

# define the evaluation loop
def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, token_type_ids, labels = batch
            logits, pred = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits.view(-1, config.num_labels), labels)
            acc = accuracy_score(labels.tolist(), pred.tolist())
            epoch_loss += loss.item()
            epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)


for epoch in range(epochs):
    train_loss, train_acc = train(model,train_loader,optimizer, criterion)
    test_loss, test_acc = evaluate(model, valid_loader, criterion)
    
    print(f"Epoch {epoch+1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
