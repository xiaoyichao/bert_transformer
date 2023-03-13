import torch
import torch.nn as nn
from transformers4token import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer, BertModel, BertConfig, DistilBertTokenizer, DistilBertModel, DistilBertConfig


# define the model
class BERTClassifier(nn.Module):
    def __init__(self, bert, config):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.class_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.relu_layer = nn.ReLU()
    
    def forward(self, encoder_embedding_dict):
        
        bert_outputs = self.bert(input_ids=encoder_embedding_dict["input_ids"], attention_mask=encoder_embedding_dict["attention_mask"], token_type_ids=encoder_embedding_dict["token_type_ids"]) #, token_type_ids=encoder_embedding_dict["token_type_ids"]
        pool_hidden_state = torch.mean(bert_outputs.last_hidden_state, dim=1)

        linear_1 = self.linear_1(pool_hidden_state)
        linear_2 = self.linear_2(linear_1)

        logits = self.class_layer(linear_2)
        pred = torch.argmax(logits, dim=1)
        return logits, pred
