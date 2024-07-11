import torch
import math
import torch.nn as nn
import torch.nn.functional as F

batch_size = 64
seq_len = 512 # F
width = 768
num_attention_heads = 12
from_seq_len = seq_len
to_seq_len = seq_len

class MultiHeadAttention(nn.Module):
    def __init__(self, width, num_attention_heads):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = width // num_attention_heads
        self.q_layer = nn.Linear(width, width)
        self.k_layer = nn.Linear(width, width)
        self.v_layer = nn.Linear(width, width)
        self.dk_sqrt = math.sqrt(self.attention_head_size)

    def transpose_for_scores(self, x, batch_size, seq_len):
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size) # [B,S,N,H]
        return x.transpose(1, 2) # [B,N,S,H]

    def forward(self, from_tensor, to_tensor, to_mask=None):
        batch_size = from_tensor.size(0)
        from_seq_len = from_tensor.size(1)
        to_seq_len = to_tensor.size(1)

        q = self.q_layer(from_tensor)
        k = self.k_layer(to_tensor)
        v = self.v_layer(to_tensor)

        q = self.transpose_for_scores(q, batch_size, from_seq_len)
        k = self.transpose_for_scores(k, batch_size, to_seq_len)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) / self.dk_sqrt
        
        if to_mask is not None:
            to_mask = to_mask.unsqueeze(1)
            attention_scores += (1.0 - to_mask) * -10000.0
        
        attention_probs = F.softmax(attention_scores, dim=-1)

        v = self.transpose_for_scores(v, batch_size, to_seq_len)
        context_layer = torch.matmul(attention_probs, v)
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, from_seq_len, self.num_attention_heads * self.attention_head_size)
        
        return context_layer

def create_attention_mask(from_tensor, to_mask):
    # B, T => B,1,T
    # B, F => B,F,1
    # B,F,1 * B,1,T =>B,F,T   
    batch_size, from_seq_len = from_tensor.size(0), from_tensor.size(1)

    to_seq_len = to_mask.size(1)
    to_mask = to_mask.view(batch_size, 1, to_seq_len)
    broadcast_ones = torch.ones(batch_size, from_seq_len, 1)
    attention_mask = broadcast_ones * to_mask
    return attention_mask

if __name__ == "__main__":
    embedding = torch.rand(batch_size, from_seq_len, width)
    to_mask = torch.rand(batch_size, from_seq_len)
    ones = torch.ones(batch_size, from_seq_len)
    zeros = torch.zeros(batch_size, from_seq_len)
    to_mask = torch.where(to_mask < 0.5, zeros, ones)
    to_mask = create_attention_mask(embedding, to_mask)
    
    attention_layer = MultiHeadAttention(width, num_attention_heads)
    y = attention_layer(embedding, embedding, to_mask)
    print(y.shape)
