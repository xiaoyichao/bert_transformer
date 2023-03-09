'''
Author: xiaoyichao xiao_yi_chao@163.com
Date: 2023-02-28 18:58:21
LastEditors: xiaoyichao xiao_yi_chao@163.com
LastEditTime: 2023-02-28 19:06:14
FilePath: /bert_transformer/bert/also_cool.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import sys
import torch.nn as nn
import torch.optim as optim
import pickle
import common4bert
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer, BertModel, BertConfig
from sklearn.metrics import accuracy_score, ndcg_score
from sklearn.model_selection import train_test_split



if torch.cuda.is_available():
    print("torch.cuda.current_device()", torch.cuda.current_device())
else:
    print("USE CPU")

data_dir_path = "/data/search_opt_model/topk_opt/rank_fine_row_cv_userprofile"
pkl_examples_limit = 120

# load tokenizer and model
# my_bert_path = "/data/search_opt_model/topk_opt/distilbert/distilbert_torch"
my_bert_path = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(my_bert_path)
config = AutoConfig.from_pretrained(my_bert_path, num_labels=4)
# config.model_type= "bert"
bert = AutoModel.from_pretrained(my_bert_path, config=config)
print(list(bert.state_dict().keys()))



class SearchDataset(Dataset):
    def __init__(self, pkl_file, max_length=128) -> None:
        with open(pkl_file, "rb") as f:
            self.data = pickle.load(f, encoding='bytes')
            if pkl_examples_limit !=-1:
                self.data = self.data[:pkl_examples_limit]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_maxlen = 32
        self.max_len = 16

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 可以在这个位置加入处理数据的逻辑
        # text = [self.data[index][2],self.data[index][3]] #,self.data[index][4]
        query = self.data[index][2]
        doc_title = self.data[index][3]
        doc_remark = self.data[index][4]

        # encoded_dict = self.tokenizer(text, text_pair, padding="max_length", truncation=True, max_length=self.max_length,return_tensors="pt")
        first_encoded_dict = tokenizer.encode_plus(
                        query,     # 输入文本
                        add_special_tokens = True, # 添加特殊标记 [CLS] 和 [SEP]
                        max_length = self.query_maxlen,           # 最大文本长度
                        pad_to_max_length = False,  # 对文本进行padding
                        truncation = True,         # 对文本进行truncation
                        return_attention_mask = True,   # 返回 attention mask
                        return_tensors = 'pt',      # 返回pytorch tensors格式的编码结果
                   )

        second_third_encoded_dict = tokenizer.encode_plus(
                        doc_title,     # 输入文本
                        doc_remark,               # 输入文本
                        add_special_tokens = True, # 添加特殊标记 [CLS] 和 [SEP]
                        max_length = (self.max_len+1)-first_encoded_dict['input_ids'].shape[-1]+1,    # 最大文本长度,这段文本的第一个cls会被删掉。
                        pad_to_max_length = True,  # 对文本进行padding
                        truncation = True,         # 对文本进行truncation
                        return_attention_mask = True,   # 返回 attention mask
                        return_tensors = 'pt',      # 返回pytorch tensors格式的编码结果
                   )
        

        
        input_ids = torch.cat((first_encoded_dict['input_ids'], second_third_encoded_dict["input_ids"][:,1:]), dim=-1)

        second_third_token_type_ids = torch.where(second_third_encoded_dict["token_type_ids"]==1,torch.tensor(2), second_third_encoded_dict["token_type_ids"])
        second_third_token_type_ids = torch.where(second_third_token_type_ids==0, torch.tensor(1), second_third_token_type_ids)
        second_third_token_type_ids = second_third_token_type_ids[:,1:]

        
        token_type_ids = torch.cat((first_encoded_dict['token_type_ids'], second_third_token_type_ids),dim=-1)
        attention_mask = torch.cat((first_encoded_dict['attention_mask'], second_third_encoded_dict['attention_mask'][:,1:]), dim=-1)
        
        label = torch.tensor(int(self.data[index][1]), dtype=torch.int)

        return input_ids, attention_mask, token_type_ids, label
    

all_pkl_names, all_pkl_paths, _ = common4bert.get_models(data_dir_path, False)
pkl_path = all_pkl_paths[-1]
dataset = SearchDataset(pkl_path)
encoding = dataset.__getitem__(0)
print(encoding, encoding)

train_size = int(0.8*dataset.__len__())
valid_size = dataset.__len__() - train_size
train_dataset, valid_dataset  = random_split(dataset,[train_size, valid_size])


batch_size = 32
train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader =  DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)



# define the model
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert = bert
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0]
        logits = self.classifier(last_hidden_state)
        return logits

model = BERTClassifier(bert)
state_dict = model.state_dict()


# define hyperparameters
epochs = 2
lr = 1e-5

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
        logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(logits.view(-1, config.num_labels), labels)
        acc = accuracy_score(labels.tolist(), logits.argmax(dim=1).tolist())
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
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(logits.view(-1, config.num_labels), labels)
            acc = accuracy_score(labels.tolist(), logits.argmax(dim=1).tolist())
            epoch_loss += loss.item()
            epoch_acc += acc
    return epoch_loss / len(loader), epoch_acc / len(loader)


for epoch in range(epochs):
    train_loss, train_acc = train(model,train_loader,optimizer, criterion)
    # test_loss, test_acc = evaluate(model, valid_loader, criterion)
    
    print(f"Epoch {epoch+1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    # print(f"\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")
