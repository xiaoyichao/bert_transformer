import torch
import math
import torch.nn as nn
import torch.nn.functional as F


batch_size = 32
seq_len = 64
width= 768
num_attention_heads = 12
attention_head_size = width//num_attention_heads



def transpose_for_scores(input_tensor, batch_size, seq_len, num_attention_heads, attention_head_size):
    input_tensor = torch.reshape(input_tensor, (batch_size, seq_len, num_attention_heads, attention_head_size))
    input_tensor = torch.transpose(input_tensor, 2, 1) # B,T,N,H -> B,N,T,H
    return input_tensor
    

def attention(from_tensor, to_tensor):
    dense_unit = 768
    q_layer = nn.Linear(dense_unit, dense_unit)
    k_layer = nn.Linear(dense_unit, dense_unit)
    v_layer = nn.Linear(dense_unit, dense_unit)

    q = q_layer(from_tensor)  # B,F,N*H
    k = k_layer(to_tensor) # B,F,N*H
    v = v_layer(to_tensor) # B,T,N*H
    
    q = transpose_for_scores(from_tensor, batch_size, seq_len, num_attention_heads, attention_head_size) # B,F,N*H ->B,N, F, H
    k = transpose_for_scores(k, batch_size, seq_len, num_attention_heads, attention_head_size) # B,T,N*H ->B,N, T, H

    attention_score = torch.matmul(q, torch.transpose(k, -1,-2)) #B,N, F, H * B,N, H,T =>B,N, F,T 
    d_sqrt =  1/math.sqrt(float(width))
    attention_score =  torch.mul(attention_score, d_sqrt)
    attention_score = F.softmax(attention_score)

    v = transpose_for_scores(v, batch_size, seq_len, num_attention_heads, attention_head_size) # B,T,N*H ->B,N, T, H
    
    y = torch.mul(attention_score, v) # B,N, F,T  * B,N,T,H ->B,N,T,H
    y = torch.transpose(y, 2,1) # B,N,T,H-> B,T,N,H
    y = torch.reshape(y, (batch_size, seq_len, num_attention_heads*attention_head_size))

    return y



if __name__ == '__main__' :
    from_tensor = torch.rand(batch_size, seq_len, width)
    y = attention(from_tensor, from_tensor)
    print(y.shape)

