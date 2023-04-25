'''
Author: xiaoyichao xiao_yi_chao@163.com
Date: 2023-02-28 18:58:21
LastEditors: xiaoyichao xiao_yi_chao@163.com
LastEditTime: 2023-02-28 19:06:14
FilePath: /bert_transformer/bert/also_cool.py
Description: deepspeed

tensorboard --logdir=bert/experiment_log

'''
import torch
import deepspeed
import sys
import os
import argparse
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)

writer = SummaryWriter('./experiment')


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_path', default='/data/search_opt_model/topk_opt/rank_fine_row_cv_userprofile', type=str, help='')
    parser.add_argument('--my_bert_path', default="./models/distilbert_torch", type=str, help='')
    parser.add_argument('--num_train_epochs', default=5, type=int, help='')
    parser.add_argument('--train_batch_size', default=2, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='')
    parser.add_argument('--output_dir', default='output_dir_lora/', type=str, help='')
    parser.add_argument('--log_steps', type=int, default=10, help='')
    parser.add_argument('--max_len', type=int, default=768, help='')
    parser.add_argument('--max_src_len', type=int, default=450, help='')
    parser.add_argument('--local_rank', type=int, default=0, help='')
    parser.add_argument('--lora_r', type=int, default=8, help='')
    parser.add_argument('--num_labels', type=int, default=3, help='')
    # parser.add_argument('--prompt_text', type=str,
    #                     default="你现在是一个信息抽取模型，请你帮我抽取出关系内容为\"性能故障\", \"部件故障\", \"组成\"和 \"检测工具\"的相关三元组，三元组内部用\"_\"连接，三元组之间用\\n分割。文本：",
    #                     help='')
    return parser.parse_args(args=[])

args = set_args()


tokenizer = BertTokenizer.from_pretrained(args.my_bert_path) 
config = AutoConfig.from_pretrained(args.my_bert_path, num_labels=args.num_labels)

config.num_labels = args.num_labels


# distilbert = MyBertModel.from_pretrained(my_bert_path, config=config)
distilbert = AutoModel.from_pretrained(args.my_bert_path, config=config)

# quantization_bit = 4
# distilbert = distilbert.quantize(quantization_bit)
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')



# data = [["客厅", 0] for _ in range(1000)]
data = [["客厅", 0], ["厨房", 0], ["卫生间", 0], ["冰箱", 1], ["洗衣机", 1], ["电视", 1],["客厅", 0], ["厨房", 0], ["卫生间", 0], ["冰箱", 0], ["洗衣机", 0], ["汽车", 2]] 
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
# scaler = GradScaler()

model = model.to(device)




conf = {"train_micro_batch_size_per_gpu": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-5,
                "betas": [
                    0.9,
                    0.95
                ],
                "eps": 1e-8,
                "weight_decay": 5e-4
            }
        },
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 1,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True
        },
        "steps_per_print": args.log_steps
        }
model_engine, optimizer, _, _ = deepspeed.initialize(config=conf,
                                                        model=model,
                                                        model_parameters=model.parameters())

print("model.state_dict().keys()", list(model.state_dict().keys()))


# define the optimizer and loss function
# optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()


# define the training loop
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch_idx, batch in enumerate(loader):

        encoder_embedding = batch
        labels = encoder_embedding["label"]

        
        logits, pred = model(encoder_embedding)
        loss = criterion(logits.view(-1, config.num_labels), labels)
        acc = accuracy_score(labels.tolist(), pred.tolist())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
    # test_loss, test_acc = evaluate(model, valid_loader, criterion)
    
    print(f"Epoch {epoch+1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    # print(f"\tValid Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
