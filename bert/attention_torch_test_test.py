import torch
import torch.nn as nn
import torch.nn.functional as F
import math

batch_size = 64
seq_len = 512
width = 768
num_attention_heads = 12
attntion_head_size = width//num_attention_heads
from_seq_len = seq_len
to_seq_len = seq_len


def transpose_scores(from_tensor, batch_size, from_seq_len, num_attention_heads, attntion_head_size):
    from_tensor = torch.reshape(
        from_tensor, (batch_size, from_seq_len, num_attention_heads, attntion_head_size))
    from_tensor = torch.transpose(from_tensor, 2, 1)  # B,F,N,H ->  B,N,F,H
    return from_tensor


def self_attention(from_tensor, to_tensor, mask=None):
    q_layer = nn.Linear(width, attntion_head_size*num_attention_heads)
    k_layer = nn.Linear(width, attntion_head_size*num_attention_heads)
    v_layer = nn.Linear(width, attntion_head_size*num_attention_heads)

    q = q_layer(from_tensor)
    k = k_layer(to_tensor)
    v = v_layer(to_tensor)

    q = transpose_scores(q, batch_size, from_seq_len,
                         num_attention_heads, attntion_head_size)  # B,N,F,H
    k = transpose_scores(k, batch_size, to_seq_len,
                         num_attention_heads, attntion_head_size)  # B,N,T,H

    attention_scores = torch.matmul(q, torch.transpose(
        k, -1, -2))  # B,N,F,H * B,N,H,T  -> B,N,F,T
    attention_scores = attention_scores*(1/math.sqrt(width))

    if mask is not None:
        adder = -(1.0-mask)*100000
        adder = torch.unsqueeze(adder, 1)
        #  B,N,F,T  + B, 1, F, T
        attention_scores = attention_scores+adder

    attention_scores = F.softmax(attention_scores, dim=-1)  # B,N,F,T

    v = torch.reshape(v, (batch_size, to_seq_len,
                      num_attention_heads, attntion_head_size))  # B,T,N,H
    v = torch.transpose(v, 1, 2)  # B,N,T,H

    y = torch.matmul(attention_scores, v)  # B,N,F,T*B,N,T,H -》B,N,F,H

    y = torch.transpose(y, 1, 2)  # B,F,N,H

    y = torch.reshape(y, (batch_size, to_seq_len,
                      num_attention_heads*attntion_head_size))  # B,F,N*H
    return y


def creat_mask(to_mask):

    to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_len))  # B,1,T
    boardcast = torch.ones(batch_size, to_seq_len, 1)  # B,F,1
    to_mask = torch.mul(boardcast, to_mask)  # B,F,1 * B,1,T -> B,F,T
    return to_mask


if __name__ == '__main__':
    emb = torch.rand(batch_size, seq_len, width)
    to_mask = torch.rand(batch_size, seq_len)
    one_mask = torch.ones(batch_size, seq_len)
    zero_mask = torch.zeros(batch_size, seq_len)
    to_mask = torch.where(to_mask >= 0.5, one_mask, zero_mask)
    to_mask = creat_mask(to_mask)
    y = self_attention(emb, emb, to_mask)
    print(y.shape)
