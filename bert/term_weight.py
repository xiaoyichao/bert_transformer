import torch
import sys
import torch.nn as nn
import torch.optim as optim
import pickle
import common4bert
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader, random_split
# from three_piece_tokenizer import ThreePieceTokenizer
from model import TermWeightModel
# from transformer_model_ import  MyBertModel
from intent_data_loader import IntentDataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer, BertModel, BertConfig
from sklearn.metrics import accuracy_score, ndcg_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast


my_bert_path = "./models/distilbert_torch"
query_path = "./data/term_weight/query.txt"
query_qieci_path = "./data/term_weight/term_weight_query_qieci.txt"
labels_path = "./data/term_weight/term_weight_labels.txt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)
writer = SummaryWriter('./experiment')

num_labels = 3
batch_size = 32
epochs = 1
lr = 1e-5
max_len = 32
max_term = 8


# 初始化Bert的参数
tokenizer = BertTokenizer.from_pretrained(my_bert_path) 
config = AutoConfig.from_pretrained(my_bert_path, num_labels=num_labels)
distil_bert = AutoModel.from_pretrained(my_bert_path)
config.max_len = max_len
config.max_term = max_term

# 读取数据
def read_data(query_path, query_qieci_path, labels_path):
    query_list = []
    query_term_list = []
    labels_list = []
    with open(query_qieci_path, 'r') as f:
        query_qieci = f.read().split("\n")[1:]
        for qieci in query_qieci:
            qieci = qieci.split(" ")
            query_term_list.append(qieci)


    with open(labels_path, 'r') as f:
        labels = f.read().split("\n")[1:]
        for label in labels:
            label = list(label)
            label = [int(i)-1 for i in label]
            labels_list.append(label)

    with open(query_path, 'r') as f:
        queries = f.read().split("\n")[1:]
        for query in queries:
            query_list.append(query)

    data = []
    assert len(query_list)==len(query_term_list) and len(query_term_list)==len(labels_list)
    for query, query_term, labels in zip(query_list, query_term_list, labels_list):
        # if len(query_term)!=len(labels):
        #     print(" len(query_term)!=len(labels)", query_term, labels)
        assert len(query_term)==len(labels) , print(" len(query_term)!=len(labels)", query_term, labels)
        data.append([query, query_term, labels])
        

    return data


data = read_data(query_path, query_qieci_path, labels_path)
              

# 数据Dataset
class TermWeightDataset(Dataset):
    def __init__(self, tokenizer, data, max_length=32, max_term=10):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.max_term = max_term

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):
        
        query = self.data[index][0]
        terms = self.data[index][1]
        labels = self.data[index][2]
        encoded_dict = self.tokenizer(query, padding = 'max_length', max_length = self.max_length, truncation=True, return_tensors='pt')
        input_ids = torch.squeeze(encoded_dict['input_ids'])
        attention_mask = torch.squeeze(encoded_dict["attention_mask"])
        token_type_ids = torch.squeeze(encoded_dict["token_type_ids"])
        labels = torch.tensor(labels, dtype=torch.float64)

        query_encoder_dict = OrderedDict()

        query_encoder_dict["input_ids"] = input_ids.to(device)
        query_encoder_dict["attention_mask"] = attention_mask.to(device)
        query_encoder_dict["token_type_ids"] = token_type_ids.to(device)
        # query_encoder_dict["label"] = labels.to(device)

        terms_encoder_dict = []
        for term in terms:
            encoded_dict = self.tokenizer(term, padding = 'max_length', max_length = self.max_length, truncation=True, return_tensors='pt')
            input_ids = torch.squeeze(encoded_dict['input_ids'])
            attention_mask = torch.squeeze(encoded_dict["attention_mask"])
            token_type_ids = torch.squeeze(encoded_dict["token_type_ids"])
            term_encoder_dict = OrderedDict()
            term_encoder_dict["input_ids"] = input_ids.to(device)
            term_encoder_dict["attention_mask"] = attention_mask.to(device)
            term_encoder_dict["token_type_ids"] = token_type_ids.to(device)
            terms_encoder_dict.append(term_encoder_dict)
        

        return query_encoder_dict, terms_encoder_dict, labels
        
    def collate_fn(self, batch):
        terms_input_ids_list = []
        terms_token_type_ids_list = []
        terms_attention_masks_list = []
        labels = []
        querys_encoder_dict = []

        for i, sample in enumerate(batch):
            query_emb = sample[0]
            terms_emb = sample[1]
            label = sample[2]
            # labels_in_batch = []
            terms_input_ids = []
            terms_token_type_ids = []
            terms_attention_masks = []
            for term_emb in terms_emb:
                terms_input_ids.append(term_emb['input_ids'])
                terms_token_type_ids.append(term_emb['token_type_ids'])
                terms_attention_masks.append(term_emb['attention_mask'])

            padding_input_ids = torch.zeros_like(term_emb['input_ids'])
            padding_token_type_ids = torch.zeros_like(
                term_emb['token_type_ids'])
            padding_attention_mask = torch.zeros_like(
                term_emb['attention_mask'])
            padding_label_tensor = torch.tensor([0.0])

            assert len(terms_emb) == len(label), print("len(terms_emb) != len(label)", len(terms_emb), len(label))
            for _ in range(self.max_term-len(terms_emb)):
                terms_input_ids.append(padding_input_ids)
                terms_token_type_ids.append(padding_token_type_ids)
                terms_attention_masks.append(padding_attention_mask)
                label = torch.cat((label, padding_label_tensor), dim=0)

            terms_input_ids_list.append(terms_input_ids)
            terms_token_type_ids_list.append(terms_token_type_ids)
            terms_attention_masks_list.append(terms_attention_masks)
            querys_encoder_dict.append(query_emb)
            labels.append(label)


        terms_encoder_dict =  {
            'input_ids': terms_input_ids_list,
            'token_type_ids': terms_token_type_ids_list,
            'attention_masks': terms_attention_masks_list,
        }
        return terms_encoder_dict, querys_encoder_dict, labels


dataset = TermWeightDataset(tokenizer=tokenizer, data=data, max_length=config.max_len, max_term=config.max_term)
encoding = dataset.__getitem__(0)
print("encoding: ", encoding)
print("pkl数据总长度: ", dataset.__len__())


train_size = int(0.8*dataset.__len__())
valid_size = dataset.__len__() - train_size
print("train数据长度: ", train_size)
print("valid数据长度: ", valid_size)
train_dataset, valid_dataset  = random_split(dataset,[train_size, valid_size])

train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
valid_loader =  DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)



# 创建模型
model = TermWeightModel(distil_bert, config)
scaler = GradScaler()
model = model.to(device)

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

        query_encoder_embedding_dict, terms_encoder_embedding_dict_list, labels = batch[0], batch[1], batch[2]
        # labels = query_encoder_embedding_dict["label"]

        # optimizer.zero_grad()
        # logits, pred = model(encoder_embedding)
        # loss = criterion(logits.view(-1, config.num_labels), labels)
        # acc = accuracy_score(labels.tolist(), pred.tolist())
        # loss.backward()
        # optimizer.step()
        # epoch_loss += loss.item()
        # epoch_acc += acc

        with autocast():
            loss = 0
            logits, preds, _ = model(query_encoder_embedding_dict, terms_encoder_embedding_dict_list, labels)
            term_len = len(labels)
            for logit, label in zip(logits, labels):
                tmp_loss = criterion(logit.view(config.num_labels), label)
                loss+=tmp_loss

        optimizer.zero_grad()
        acc = accuracy_score(labels.tolist(), preds.tolist())
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
    # test_loss, test_acc = evaluate(model, valid_loader, criterion)
    
    print(f"Epoch {epoch+1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    # print(f"\tValid Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

