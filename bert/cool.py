import torch 
import torch.nn as nn
import torch.optim as optim
import torchtext.datasets as datasets
# from torchtext import data, datasets
# from torchtext.legacy.data import Field,TabularDataset,Iterator,BucketIterator,LabelField
from transformers import AutoModel, AutoTokenizer, AutoConfig

# 定义模型和tokenizer
model_name = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name, num_labels=2)
model = AutoModel.from_pretrained(model_name,config=config)

#define fields（字段） and dataset

def tokenize(label, line):
    return tokenizer(line)

#加载数据
train_data, test_date = datasets.SogouNews(split=("train","test"))


trainloader = torch.utils.data.DataLoader(train_data, batch_size=4,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(test_date, batch_size=4,
                                          shuffle=True, num_workers=2)

#创建词表
# LABEL.build_vocab(train_data)

BATCH_SIZE = 32

# create data loaders
train_iterator, test_iterator = data.BucketIterator.split((train_data, test_date), batch_size=BATCH_SIZE, device="cuda")

# define model
class BERTClassifier(nn.Module):
    def __init__(self,bert,num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = nn.Linear(config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

# define train loop   
def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:
        input_ids= batch.text
        attention_mask = (input_ids!=0)
        labels = batch.label
         
        optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask)

        loss = criterion(logits.view(-1, len(LABEL.vocab)), labels)

        acc = ((logits.argmax(dim=1) == labels).sum())/len(labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch.text
            attention_mask = (input_ids!=0)
            labels = batch.label
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(logits.view(-1,len(LABEL.vocab)), labels)

            acc = ((logits.argmax(dim=1) == labels).sum())/len(labels)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss/len(iterator), epoch_acc/len(iterator)

N_EPOCHS = 2
LR = 1e-5
criterion = nn.CrossEntropyLoss()

for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model,train_iterator,criterion)
    test_loss, test_acc = evaluate(model, test_iterator, criterion)
    
    print(f"Epoch {epoch+1}")
    print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
    print(f"\tTest Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")




        

