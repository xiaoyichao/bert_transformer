import torch
import math
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64
width = 768
seq_len = 512
head_num = 12

class SelfAttention(nn.Module):
    def __init__(self, batch_size, seq_len, width, head_num):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.width = width
        self.head_num = head_num
        self.head_size = self.width//head_num
        self.q_layer = nn.Linear(self.width, self.width)
        self.k_layer = nn.Linear(self.width, self.width)
        self.v_layer = nn.Linear(self.width, self.width)
        self.dk_sqrt = math.sqrt(self.head_size)

    def transpose_for_score(self, v):
        v = v.view(self.batch_size, self.seq_len, self.head_num, self.head_size)
        return v.transpose(1,2)

    def forward(self, from_tensor, to_tensor, mask=None):
        q = self.q_layer(from_tensor)
        k = self.k_layer(to_tensor)
        v = self.v_layer(to_tensor)

        q = self.transpose_for_score(q)
        k = self.transpose_for_score(k)

        attention_score = torch.matmul(q, k.transpose(-1,-2))/self.dk_sqrt
        if mask is not None:
            mask = mask.unsqueeze(1)
            attention_score = attention_score + (1.0 -mask)*(-1e5)

        attention_score = F.softmax(attention_score, dim=-1)
        
        v = self.transpose_for_score(v)

        attention_pros = torch.matmul(attention_score, v)
        attention_pros = attention_pros.transpose(1,2).contiguous()
        output = attention_pros.view(self.batch_size, self.seq_len, self.head_num*self.head_size)
        return output

def creat_attention_mask(from_tensor, to_mask):
    batch_size = from_tensor.size(0)
    from_seq_len = from_tensor.size(1)
    to_seq_len = to_mask.size(1)
    to_mask = to_mask.view(batch_size, 1, to_seq_len)
    boardcast_ones = torch.ones(batch_size, from_seq_len, 1)
    return boardcast_ones*to_mask

from_tensor = torch.rand(64,512,768)
to_mask = torch.rand(64,512)
ones = torch.ones(64,512)
zeros = torch.zeros(64,512)
to_mask = torch.where(to_mask<0.5, zeros, ones)
attention_mask = creat_attention_mask(from_tensor, to_mask)
attention = SelfAttention(64,512,768,12)
y= attention(from_tensor, from_tensor, attention_mask)
print(y.shape)







        

