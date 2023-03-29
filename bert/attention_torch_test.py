import torch
import math
import torch.nn as nn
import torch.nn.functional as F


batch_size = 32
seq_len = 64
width= 768
num_attention_heads = 12
attention_head_size = width//num_attention_heads
from_seq_length = seq_len
to_seq_length = seq_len


def transpose_for_scores(input_tensor, batch_size, seq_len, num_attention_heads, attention_head_size):
    input_tensor = torch.reshape(input_tensor, (batch_size, seq_len, num_attention_heads, attention_head_size))
    input_tensor = torch.transpose(input_tensor, 2, 1) # B,T,N,H -> B,N,T,H
    return input_tensor
    

def create_attention_mask(from_seq_len, to_mask):
    '''
    to_mask (batch_size, from_seq_len)
    0 是要被mask的位置
    '''
    # [B,T]-> [B,1,T] 
    # [B,F] -> [B, F, 1]
    # [B, F, 1] [B,1,T]  -> [B,F,T] 
    to_seq_len = to_mask.shape[1]
    to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_len))

    broadcast_ones = torch.ones(batch_size, from_seq_len, 1).to(torch.float32)
    atten_mask = torch.mul(broadcast_ones, to_mask)
    return atten_mask



def attention(from_tensor, to_tensor, to_mask=None):
    
    from_tensor = torch.reshape(from_tensor,(-1, width))
    to_tensor = torch.reshape(to_tensor, (-1, width))
    q_layer = nn.Linear(width, num_attention_heads * attention_head_size)
    k_layer = nn.Linear(width, num_attention_heads * attention_head_size)
    v_layer = nn.Linear(width, num_attention_heads * attention_head_size)

    q = q_layer(from_tensor)  # B*F,N*H
    k = k_layer(to_tensor) # B*F,N*H
    v = v_layer(to_tensor) # B*T,N*H
    
    q = transpose_for_scores(q, batch_size, from_seq_length, num_attention_heads, attention_head_size) # B*F,N*H ->B,N, F, H
    k = transpose_for_scores(k, batch_size, to_seq_length, num_attention_heads, attention_head_size) # B*T,N*H ->B,N, T, H

    attention_score = torch.matmul(q, torch.transpose(k, -1,-2)) #B,N, F, H * B,N, H,T =>B,N, F,T 
    d_sqrt = 1/math.sqrt(width)
    attention_score = torch.mul(attention_score, d_sqrt)
    if to_mask is not None:
        atten_mask = create_attention_mask(from_seq_length, to_mask)
        atten_mask = torch.unsqueeze(atten_mask, 1) #[B,F,T]->[B,1,F,T]
        adder = (1-atten_mask) * -100000.0
        attention_score += adder
        
    attention_score = F.softmax(attention_score,dim=-1)

    # B,T,N*H ->B,N, T, H
    v = torch.reshape(v, (batch_size, to_seq_length, num_attention_heads, attention_head_size)) 
    v = torch.transpose(v, 2, 1)
    
    y = torch.mul(attention_score, v) # B,N, F,T  * B,N,T,H ->B,N,T,H
    y = torch.transpose(y, 2,1) # B,N,T,H-> B,T,N,H
    y = torch.reshape(y, (batch_size, from_seq_length, num_attention_heads*attention_head_size))

    return y



if __name__ == '__main__' :

    from_tensor = torch.rand(batch_size, from_seq_length, width)

    to_mask = torch.rand(batch_size, from_seq_length)
    ones = torch.ones(batch_size, from_seq_length)
    zero = torch.zeros(batch_size, from_seq_length)
    to_mask = torch.where(to_mask<0.5, zero, ones)
    
    y = attention(from_tensor, from_tensor, to_mask)
    print(y.shape)

