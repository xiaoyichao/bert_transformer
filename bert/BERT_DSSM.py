import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import random

class BertDSSM(nn.Module):
    def __init__(self, bert_model_path):
        super(BertDSSM, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path)
        self.fc = nn.Linear(self.bert.config.hidden_size, 128)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        embediding = self.relu(self.fc(pooled_output))
        return embediding

def hinge_loss(pos_scores, neg_scores, margin=1.0):
    # loss = torch.mean(torch.max(torch.tensor(0.0).to(pos_scores.device), margin - pos_scores + neg_scores))
    loss = torch.mean(torch.max(torch.zeros_like(pos_scores), torch.ones_like(pos_scores) + neg_scores - pos_scores))
    return loss



def batch_hard_negative_sampling(batch_data, tokenizer, bert_model_path, device='cuda'):
    model = BertDSSM(bert_model_path).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    for batch in batch_data:
        queries, pos_docs, neg_docs = batch

        # Tokenize queries, positive docs, and negative docs
        queries_enc = tokenizer(queries, padding=True, truncation=True, return_tensors='pt')
        pos_docs_enc = tokenizer(pos_docs, padding=True, truncation=True, return_tensors='pt')
        neg_docs_enc = tokenizer(neg_docs, padding=True, truncation=True, return_tensors='pt')

        # Move to the device
        for key in queries_enc:
            queries_enc[key] = queries_enc[key].to(device)
            pos_docs_enc[key] = pos_docs_enc[key].to(device)
            neg_docs_enc[key] = neg_docs_enc[key].to(device)

        # Calculate scores
        query_embedding = model(input_ids=queries_enc['input_ids'], attention_mask=queries_enc['attention_mask'])
        pos_embedding = model(input_ids=pos_docs_enc['input_ids'], attention_mask=pos_docs_enc['attention_mask'])
        neg_embedding = model(input_ids=neg_docs_enc['input_ids'], attention_mask=neg_docs_enc['attention_mask'])
        pos_scores = F.cosine_similarity(query_embedding, pos_embedding)
        neg_scores= F.cosine_similarity(query_embedding, neg_embedding)
        # Calculate loss

        new_neg_embedding = []
        new_pos_embedding = []
        new_query_embedding =[]

        loss = hinge_loss(pos_scores, neg_scores)
        
        for i in range(len(queries)-1):
            # Randomly select a negative document
            if not torch.all(torch.eq(queries_enc["input_ids"][i], queries_enc["input_ids"][i+1])):
                new_neg_embedding.append(pos_embedding[i+1])
                new_pos_embedding.append(pos_embedding[i])
                new_query_embedding.append(query_embedding[i])

        new_query_embedding = torch.stack(new_query_embedding).reshape(-1, 128)
        new_pos_embedding = torch.stack(new_pos_embedding).reshape(-1, 128)
        new_neg_embedding = torch.stack(new_neg_embedding).reshape(-1, 128)
        pos_scores = F.cosine_similarity(new_query_embedding, new_pos_embedding)
        neg_scores= F.cosine_similarity(new_query_embedding, new_neg_embedding)
        # Calculate loss
        loss = (loss + hinge_loss(pos_scores, neg_scores))/2

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")


def main():
    bert_model_path = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Prepare some dummy data for demonstration purposes
    batch_data = [(["客厅", "厨房", "洗手间", "客厅", "洗手间", "踢脚线", "绿植", "沙发"],
                   ["客厅", "厨房", "洗手间", "客厅", "洗手间", "踢脚线", "绿植", "沙发"],
                   ["厨房", "客厅", "绿植", "厨房", "客厅", "沙发", "日式", "新中式"]
                   )
                  ]

    batch_hard_negative_sampling(batch_data, tokenizer, bert_model_path, device)



if __name__ == "__main__":
    main()

