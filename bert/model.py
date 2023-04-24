import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers4token import AutoModel, AutoTokenizer, AutoConfig, BertTokenizer, BertModel, BertConfig, DistilBertTokenizer, DistilBertModel, DistilBertConfig



class TermWeightModelSample(nn.Module):
    def __init__(self, distilbert, config):
        super(TermWeightModelSample, self).__init__()
        self.distilbert = distilbert
        self.linear_1 = nn.Linear(1, 16)
        self.class_layer = nn.Linear(16, config.num_labels)
        nn.init.xavier_uniform_(self.class_layer.weight)
        self.config = config

        self.relu_layer = nn.GELU()
        
        # 冻结 DistilBERT 参数
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # 解冻顶层参数
        # for param in self.distilbert.encoder.layer[-1].parameters():
        #     param.requires_grad = True

        # 确认 BERT 参数被冻结
        for name, param in self.distilbert.named_parameters():
            if param.requires_grad is False:
                print("确认 BERT 参数被冻结", name, param.requires_grad)

    def forward(self, query_encoder_embedding_dict, terms_encoder_embedding_dict_list, labels=None):
        logits = []
        preds = []
        if labels is not None:
            labels = labels.to(torch.long)

        query_bert_outputs = self.distilbert(input_ids=query_encoder_embedding_dict["input_ids"], attention_mask=query_encoder_embedding_dict["attention_mask"], token_type_ids=query_encoder_embedding_dict["token_type_ids"])
        query_emb = torch.mean(query_bert_outputs.last_hidden_state, dim=1)

        cos_sims = []
        for term_encoder_embedding_dict in terms_encoder_embedding_dict_list:
            term_bert_outputs = self.distilbert(input_ids=term_encoder_embedding_dict["input_ids"], attention_mask=term_encoder_embedding_dict["attention_mask"], token_type_ids=term_encoder_embedding_dict["token_type_ids"])
            term_emb = torch.mean(term_bert_outputs.last_hidden_state, dim=1)
            cos_sim = F.cosine_similarity(query_emb, term_emb)
            if cos_sim>=0.8:
                cos_sims.append(2.0)
            elif cos_sim>=0.5:
                cos_sims.append(1.0)
            else:
                cos_sims.append(0.0)

            # linear_1 = self.relu_layer(self.linear_1(cos_sim))
            # logit = self.class_layer(linear_1)
            # pred = torch.argmax(logit, dim=0)
            # logits.append(logit)
            # preds.append(pred)
        
        # padded_cos_sims = torch.nn.utils.rnn.pad_sequence(cos_sims, batch_first=True, padding_value=-1)
        # padded_cos_sims = torch.nn.functional.pad(torch.tensor(cos_sims), (-1, self.config.max_term-len(cos_sims)+1), mode='constant', value=-1)

        # linear_1 = self.relu_layer(self.linear_1(padded_cos_sims))
        # logit = self.class_layer(linear_1)
        # pred = torch.argmax(logit, dim=0)
            # logits.append(logit)
            # preds.append(pred)

        return torch.unsqueeze(torch.tensor(cos_sims), 0), torch.unsqueeze(torch.tensor(cos_sims), 0), labels


class TermWeightModel(nn.Module):
    def __init__(self, distilbert, config):
        super(TermWeightModel, self).__init__()
        self.distilbert = distilbert
        self.config = config
        self.linear_1 = nn.Linear(self.config.max_term_num, self.config.max_term_num)
        self.class_layer = nn.Linear(1, config.num_labels)
        nn.init.xavier_uniform_(self.class_layer.weight)
        

        self.relu_layer = nn.GELU()
        
        # 冻结 DistilBERT 参数
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # 解冻顶层参数
        # for param in self.distilbert.encoder.layer[-1].parameters():
        #     param.requires_grad = True

        # 确认 BERT 参数被冻结
        for name, param in self.distilbert.named_parameters():
            if param.requires_grad is False:
                print("确认 BERT 参数被冻结", name, param.requires_grad)

    def forward(self, query_encoder_embedding_dict, terms_encoder_dict=None):
        # logits = []
        # preds = []

        query_bert_outputs = self.distilbert(input_ids=query_encoder_embedding_dict["input_ids"], attention_mask=query_encoder_embedding_dict["attention_mask"], token_type_ids=query_encoder_embedding_dict["token_type_ids"]) # [batch_size, query_seq_len, emb_size]
        query_emb = torch.mean(query_bert_outputs.last_hidden_state, dim=1) # [batch_size, emb_size]
        query_emb = query_emb.repeat(self.config.max_term_num, 1) # [batch_size*term_size, emb_size]


        terms_input_ids = torch.flatten(terms_encoder_dict['input_ids'],  start_dim=0, end_dim=1)
        terms_attention_mask = torch.flatten(terms_encoder_dict['attention_mask'],  start_dim=0, end_dim=1)
        terms_token_type_ids = torch.flatten(terms_encoder_dict['token_type_ids'],  start_dim=0, end_dim=1)
        term_bert_outputs = self.distilbert(input_ids=terms_input_ids, attention_mask=terms_attention_mask, token_type_ids=terms_token_type_ids) #[batch_size*term_size, term_seq_len, emb_size]
        term_emb = torch.mean(term_bert_outputs.last_hidden_state, dim=1) # [batch_size*term_size, emb_size]

        cos_sims = F.cosine_similarity(query_emb, term_emb)
        # cos_sims = torch.unsqueeze(cos_sims, -1) # [batch_size*term_size, 1]
        cos_sims = torch.reshape(cos_sims, (-1, self.config.max_term_num)) # [batch_size, term_size]
    
        linear_1 = self.relu_layer(self.linear_1(cos_sims))
        linear_1 = torch.reshape(linear_1,(-1,1))

        logits = self.class_layer(linear_1)
        preds = torch.argmax(logits, dim=1)

        # preds = []
        # for cos_sim in cos_sims:
        #     cos_sim = cos_sim.tolist()[0]
        #     if cos_sim>=0.8:
        #         preds.append(2.0)
        #     elif cos_sim>=0.5:
        #         preds.append(1.0)
        #     else:
        #         preds.append(0.0)

    
        return logits, preds


class DistilBERTIntent(nn.Module):
    def __init__(self, distilbert, config):
        super(DistilBERTIntent, self).__init__()
        self.distilbert = distilbert
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        # self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        # self.linear_3 = nn.Linear(config.hidden_size // 2, config.hidden_size // 4)
        self.class_layer = nn.Linear(config.hidden_size, config.num_labels)
        nn.init.xavier_uniform_(self.class_layer.weight)

        self.relu_layer = nn.GELU()
        
        # for name, param in self.distilbert.named_parameters():
        #     print(name, param.shape)
        #     if "encoder.layer.1" not in name: 
        #         param.requires_grad = False

        # 冻结 DistilBERT 参数
        for param in self.distilbert.parameters():
            param.requires_grad = False

        # 解冻顶层参数
        # for param in self.distilbert.encoder.layer[-1].parameters():
        #     param.requires_grad = True

        # 确认 BERT 参数被冻结
        for name, param in self.distilbert.named_parameters():
            if param.requires_grad is False:
                print("确认 BERT 参数被冻结", name, param.requires_grad)

    def forward(self, encoder_embedding_dict):
        bert_outputs = self.distilbert(input_ids=encoder_embedding_dict["input_ids"], attention_mask=encoder_embedding_dict["attention_mask"], token_type_ids=encoder_embedding_dict["token_type_ids"])
        pool_hidden_state = torch.mean(bert_outputs.last_hidden_state, dim=1)

        linear_1 = self.relu_layer(self.linear_1(pool_hidden_state))
        # linear_2 = self.relu_layer(self.linear_2(linear_1))
        # linear_3 = self.relu_layer(self.linear_3(linear_2))
        logits = self.class_layer(linear_1)

        pred = torch.argmax(logits, dim=1)
        return logits, pred


class DistilBERTClassifier(nn.Module):
    def __init__(self, distilbert, config):
        super(DistilBERTClassifier, self).__init__()
        self.distilbert = distilbert
        self.linear_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.class_layer = nn.Linear(config.hidden_size, config.num_labels)
        self.relu_layer = nn.ReLU()
    
    def forward(self, encoder_embedding_dict):
        
        bert_outputs = self.distilbert(input_ids=encoder_embedding_dict["input_ids"], attention_mask=encoder_embedding_dict["attention_mask"], token_type_ids=encoder_embedding_dict["token_type_ids"]) #, token_type_ids=encoder_embedding_dict["token_type_ids"]
        pool_hidden_state = torch.mean(bert_outputs.last_hidden_state, dim=1)

        linear_1 = self.relu_layer(self.linear_1(pool_hidden_state))
        linear_2 = self.relu_layer(self.linear_2(linear_1))
        
        logits = self.class_layer(linear_2)
        pred = torch.argmax(logits, dim=1)
        return logits, pred


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

        linear_1 = self.relu_layer(self.linear_1(pool_hidden_state))
        linear_2 = self.relu_layer(self.linear_2(linear_1))

        logits = self.class_layer(linear_2)
        pred = torch.argmax(logits, dim=1)
        return logits, pred
