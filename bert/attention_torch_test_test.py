import torch
import math
import torch.nn as nn
import torch.nn.functional as F


batch_size = 64
seq_len = 512
width = 768
num_attention_heads = 12
attntion_head_size = width//num_attention_heads
from_seq_len = seq_len
to_seq_len = seq_len

def transpose_for_score(from_tensor, batch_size, from_seq_len, num_attention_heads, attntion_head_size):
    from_tensor = torch.reshape(from_tensor, (batch_size, from_seq_len, num_attention_heads, attntion_head_size))
    output_tensor = torch.transpose(from_tensor, 1,2)
    return output_tensor


def attention(from_tensor, to_tensor, to_mask=None):
    from_tensor = torch.reshape(from_tensor, (-1, width)) #[B*F, N*H]
    to_tensor = torch.reshape(to_tensor, (-1, width)) 

    unit_num = num_attention_heads*attntion_head_size
    q_layer = nn.Linear(width, unit_num)#[B*F, N*H]
    k_layer = nn.Linear(width, unit_num)
    v_layer = nn.Linear(width, unit_num)

    q = q_layer(from_tensor) #[B*F, N*H]
    k = k_layer(to_tensor)
    v = v_layer(to_tensor) #[B*T, N*H]

    q = transpose_for_score(q,  batch_size, from_seq_len, num_attention_heads, attntion_head_size) #[B,N, F,H]
    k = transpose_for_score(k,  batch_size, to_seq_len, num_attention_heads, attntion_head_size)#[B,N, T,H]

    attention_score = torch.matmul(q, torch.transpose(k, -1, -2))#[B,N, F,H] * [B,N, H,T] => [B,N,F,T]
    d_sqrt = 1/math.sqrt(width)
    attention_score = torch.mul(attention_score, d_sqrt)
    
    if to_mask is not None: #
        adder = -(1-to_mask)*100000
        # B,T,F => B,1,T,F
        adder = torch.unsqueeze(adder, 1)

        # [B,N,F,T] + B,1,T,F
        attention_score = attention_score+adder
        
    attention_score = F.softmax(attention_score)


    v = torch.reshape(v, (batch_size, to_seq_len, num_attention_heads, attntion_head_size)) #[B, T, N, H]
    v = torch.transpose(v, 2, 1)#[B,N,T,H]
    y = torch.matmul(attention_score, v)#[B,N,F,T]* [B,N,T,H] => [B,N,F,H]

    y = torch.transpose(y, 1,2) # [B,F,N,H]
    y = torch.reshape(y, (batch_size, from_seq_len, num_attention_heads*attntion_head_size))

    return y


def create_attention_mask(from_tensor, to_mask):
    # B, T => B,T,1
    # B, F => B,1,F
    # B,T,1 * B,1,F =>B,T,F 

    batch_size = from_tensor.shape[0]
    from_seq_len = from_tensor.shape[1]
    to_seq_len = to_mask.shape[1]
    to_mask = torch.reshape(to_mask, (batch_size, to_seq_len, 1))
    boardcast_ones = torch.ones(batch_size, 1, from_seq_len)
    attention_mask = to_mask*boardcast_ones
    return attention_mask

    



if __name__ == "__main__":
    embedding = torch.rand(batch_size, from_seq_len, width)
    to_mask = torch.rand(batch_size, from_seq_len)
    ones =  torch.ones(batch_size, from_seq_len)
    zeros = torch.zeros(batch_size, from_seq_len)
    to_mask = torch.where(to_mask<0.5, zeros, ones)
    to_mask = create_attention_mask(embedding, to_mask)
    y = attention(embedding, embedding, to_mask)
    print(y.shape)







