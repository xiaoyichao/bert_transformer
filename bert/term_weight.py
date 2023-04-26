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
from sklearn.metrics import f1_score



my_bert_path = "./models/distilbert_torch"
query_path = "./data/term_weight/query.txt"
query_qieci_path = "./data/term_weight/term_weight_query_qieci.txt"
labels_path = "./data/term_weight/term_weight_labels.txt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)
writer = SummaryWriter('./experiment')

num_labels = 3
batch_size = 32
epochs = 100
lr = 1e-6
query_max_len = 32
term_max_len = 8
max_term_num = 8


# 初始化Bert的参数
tokenizer = BertTokenizer.from_pretrained(my_bert_path) 
config = AutoConfig.from_pretrained(my_bert_path, num_labels=num_labels)
distil_bert = AutoModel.from_pretrained(my_bert_path)
config.query_max_len = query_max_len
config.term_max_len = term_max_len
config.max_term_num = max_term_num


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
    def __init__(self, tokenizer, data, query_max_len=32, term_max_len=8, max_term_num=10):
        self.tokenizer = tokenizer
        self.data = data
        self.query_max_len = query_max_len
        self.term_max_len = term_max_len
        self.max_term_num = max_term_num

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, index):
        
        query = self.data[index][0]
        terms = self.data[index][1]
        labels = self.data[index][2]
        encoded_dict = self.tokenizer(query, padding = 'max_length', max_length = self.query_max_len, truncation=True, return_tensors='pt')
        input_ids = torch.squeeze(encoded_dict['input_ids'])
        attention_mask = torch.squeeze(encoded_dict["attention_mask"])
        token_type_ids = torch.squeeze(encoded_dict["token_type_ids"])
        labels = torch.tensor(labels, dtype=torch.int64).to(device)

        query_encoder_dict = OrderedDict()

        query_encoder_dict["input_ids"] = input_ids.to(device)
        query_encoder_dict["attention_mask"] = attention_mask.to(device)
        query_encoder_dict["token_type_ids"] = token_type_ids.to(device)

        terms_encoder_dict = []
        for term in terms:
            encoded_dict = self.tokenizer(term, padding = 'max_length', max_length = self.term_max_len, truncation=True, return_tensors='pt')
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

        terms_encoder_dict_list = []
        labels = []
        masks = []
        
        queries_input_ids_list = []
        queries_token_type_ids_list = []
        queries_attention_masks_list = []

        for i, sample in enumerate(batch):
            query_emb = sample[0]
            terms_emb = sample[1]
            label = sample[2]
            # labels_in_batch = []
            terms_input_ids = []
            terms_token_type_ids = []
            terms_attention_masks = []
            mask = []
            for term_emb in terms_emb:
                terms_input_ids.append(term_emb['input_ids'])
                terms_token_type_ids.append(term_emb['token_type_ids'])
                terms_attention_masks.append(term_emb['attention_mask'])
                mask.append(torch.tensor([1]))

            padding_input_ids = torch.zeros_like(term_emb['input_ids']).to(device)
            padding_token_type_ids = torch.zeros_like(
                term_emb['token_type_ids']).to(device)
            padding_attention_mask = torch.zeros_like(
                term_emb['attention_mask']).to(device)
            # padding_label_tensor = torch.tensor([0]).to(device)
            padding_label_tensor = torch.tensor([-1]).to(device)


            assert len(terms_emb) == len(label), print("len(terms_emb) != len(label)", len(terms_emb), len(label))
            for _ in range(self.max_term_num-len(terms_emb)):
                terms_input_ids.append(padding_input_ids)
                terms_token_type_ids.append(padding_token_type_ids)
                terms_attention_masks.append(padding_attention_mask)
                label = torch.cat((label, padding_label_tensor), dim=0)
                mask.append(torch.tensor([0]))

            terms_input_ids = torch.stack(terms_input_ids, dim=0)
            terms_token_type_ids = torch.stack(terms_token_type_ids, dim=0)
            terms_attention_masks = torch.stack(terms_attention_masks, dim=0)
            mask = torch.stack(mask, dim=0)

            # term_encoder_dict = {
            # 'input_ids': terms_input_ids,
            # 'token_type_ids': terms_token_type_ids,
            # 'attention_mask': terms_attention_masks,
            # }    

            # terms_encoder_dict_list.append(term_encoder_dict)      

            terms_input_ids_list.append(terms_input_ids)
            terms_token_type_ids_list.append(terms_token_type_ids)
            terms_attention_masks_list.append(terms_attention_masks)

            queries_input_ids_list.append(query_emb['input_ids'])
            queries_token_type_ids_list.append(query_emb['token_type_ids'])
            queries_attention_masks_list.append(query_emb['attention_mask'])

            masks.append(mask)
            labels.append(label)

        queries_input_ids_list = torch.stack(queries_input_ids_list, dim=0)
        queries_token_type_ids_list = torch.stack(queries_token_type_ids_list, dim=0)
        queries_attention_masks_list = torch.stack(queries_attention_masks_list, dim=0)

        terms_input_ids_list = torch.stack(terms_input_ids_list, dim=0)
        terms_token_type_ids_list = torch.stack(terms_token_type_ids_list, dim=0)
        terms_attention_masks_list = torch.stack(terms_attention_masks_list, dim=0)

        labels = torch.stack(labels, dim=0).to(device)
        masks = torch.stack(masks, dim=0).to(device)
        masks = torch.squeeze(masks)

        queries_encoder_dict =  {
            'input_ids': queries_input_ids_list,
            'token_type_ids': queries_token_type_ids_list,
            'attention_mask': queries_attention_masks_list,
        }
    
        terms_encoder_dict =  {
            'input_ids': terms_input_ids_list,
            'token_type_ids': terms_token_type_ids_list,
            'attention_mask': terms_attention_masks_list,
        }
        
        return queries_encoder_dict, labels, terms_encoder_dict, masks


dataset = TermWeightDataset(tokenizer=tokenizer, data=data, query_max_len=config.query_max_len, term_max_len=config.term_max_len, max_term_num=config.max_term_num)
encoding = dataset.__getitem__(0)
print("encoding: ", encoding)
print("pkl数据总长度: ", dataset.__len__())


train_size = int(0.9*dataset.__len__())
valid_size = dataset.__len__() - train_size
print("train数据长度: ", train_size)
print("valid数据长度: ", valid_size)
train_dataset, valid_dataset  = random_split(dataset,[train_size, valid_size])

train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
valid_loader =  DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)



# 创建模型
model = TermWeightModel(distil_bert, config)


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
# criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(ignore_index=-1)


# define the training loop
def train(model, loader, optimizer, criterion, epoch):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for batch_idx, batch in enumerate(loader):

        query_encoder_embedding_dict, labels, terms_encoder_dict, masks  = batch[0], batch[1], batch[2], batch[3]

        with autocast():
            logits, preds = model(query_encoder_embedding_dict, terms_encoder_dict)
            labels = labels.view(-1)
            masks = masks.view(-1)
            # loss = criterion(logits, labels)
            loss = torch.tensor(0.0).to(device)
            target_for_loss = torch.where(labels != -1, labels, torch.zeros_like(labels))
            for logit, pred, label, mask in zip(logits, preds, labels, masks):
                if mask !=0:
                    loss += criterion(logit, label)
                else:
                    pass
                    
            loss = loss / masks.sum()

        
        acc = accuracy_score(labels.tolist(), preds.tolist())
        acc = (acc * masks).sum() / masks.sum()
        # acc = accuracy_score(labels.tolist(), preds)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        epoch_acc += acc
            
        # ...log the running loss
        writer.add_scalar('training loss', loss.item(), len(loader) * epoch + batch_idx)

        writer.add_scalar('training acc', acc, len(loader) * epoch + batch_idx)

    return epoch_loss / len(loader), epoch_acc / len(loader)

# define the evaluation loop
from sklearn.metrics import f1_score

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            query_encoder_embedding_dict, labels, terms_encoder_dict  = batch[0], batch[1], batch[2]
            
            logits, preds = model(query_encoder_embedding_dict, terms_encoder_dict)
            labels = labels.view(-1)
            loss = criterion(logits.view(-1 ,config.num_labels), labels)

            acc = accuracy_score(labels.tolist(), preds.tolist())
            f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
            
            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_f1 += f1
            
            writer.add_scalar('valid loss', loss.item(), len(loader) * epoch + batch_idx)
            writer.add_scalar('valid acc', acc, len(loader) * epoch + batch_idx)
            writer.add_scalar('valid f1', f1, len(loader) * epoch + batch_idx)
            
    return epoch_loss / len(loader), epoch_acc / len(loader), epoch_f1 / len(loader)



for epoch in range(epochs):
    train_loss, train_acc = train(model,train_loader,optimizer, criterion, epoch)
    test_loss, test_acc, f1 = evaluate(model, valid_loader, criterion)
    
    print(f"Epoch {epoch+1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\tValid Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%", f1)

