'''
Author: xiaoyichao xiaoyichao@haohaozhu.com
Date: 2023-01-29 14:04:06
LastEditors: root root@haohaozhu.com
LastEditTime: 2023-02-02 16:44:57
FilePath: 
Description: attention 核心功能测试
'''
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf


width = 768
batch_size =16
seq_length =128
num_attention_heads =12
attention_head_size = int(width/num_attention_heads)

from_seq_length = seq_length
to_seq_length = seq_length
embedding = torch.rand(batch_size,from_seq_length,width)
to_mask = torch.rand(batch_size,from_seq_length)

one = torch.ones_like(to_mask)   #生成与a大小一致的值全部为1的矩阵
zero = torch.zeros_like(to_mask)
to_mask = torch.where(to_mask <0.5, zero, one) #0.5为阈值
    


def transpose_for_scores(input_tensor):
    output_tensor = torch.reshape(input_tensor,(batch_size,seq_length,num_attention_heads, attention_head_size))
    # output_tensor = torch.permute(output_tensor,(0,2,1,3))
    output_tensor = torch.transpose(output_tensor,2,1)
    return output_tensor

def create_attention_mask(from_seq_length, to_mask):
    # 二维扩增到三维，在不考虑batch的条件下，实际上是把seq_length的数据复制了seq_length次，构成了这个矩阵
    to_seq_length = to_mask.shape[1]
    to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)) # [B,T] ->[B,1,T]
    to_mask = to_mask.to(dtype=torch.float32)

    broadcast_ones = torch.ones(
      (batch_size, from_seq_length, 1)).to(dtype=torch.float32)# [B,F,1]

    mask = broadcast_ones * to_mask # [B,F,1] * [B,1,T] -> [B,F,T] 
    return mask


def attention_layer(from_tensor, to_tensor, attention_mask=None):
    """attention_layer的功能实现

    Args:
        from_tensor (float Tensor of shape [batch_size, from_seq_length,from_width]): [from_tensor和from_tensor实际上使用的embedding层的输出]
        to_tensor (float Tensor of shape [batch_size, to_seq_length,from_width]): [from_tensor和from_tensor实际上使用的embedding层的输出]
        attention_mask ([B,F,T] , optional): [batch内是mask序列，1表示没有被mask,0表示被mask]. Defaults to None.

    Returns:
        [type]: [description]
    """
    dense_unit = 768
    q_layer = nn.Linear(dense_unit, dense_unit)
    k_layer = nn.Linear(dense_unit, dense_unit)
    v_layer = nn.Linear(dense_unit, dense_unit)

    q = q_layer(from_tensor) # [B,F,N*H]
    k = k_layer(to_tensor) # [B,T,N*H]
    v = v_layer(to_tensor) # [B,T,N*H]

    q = transpose_for_scores(q) # [B,F,N,H]->[B,N,F,H]
    k = transpose_for_scores(k) #[B,T,N,H]->[B,N,T,H]
    

    attention_scores = torch.matmul(q, torch.transpose(k, 2, 3)) # [B,N,F,H]* [B,N,H,T] -> [B,N,F,T]
    d_sqrt =  1/math.sqrt(float(width))
    if attention_mask is not None:
        attention_mask = torch.unsqueeze(attention_mask, 1) #[B, F, T] -> [B, 1, F, T] 扩增了纬度，这里边的元素只有0和1两种，mask的标识是0，需要注意力的标识是1。
        adder = (1-attention_mask.to(torch.float32)) * -10000.0
        # 有注意力的位置,adder = 0 , mask的位置，adder= -10000.0
        # mask的位置，在注意力分数的基础上加一个很大的负数，做softMax的时候就是接近0，也就是那些需要mask的位置，加0的相当于没操作，也就是那些没有被mask的位置，是正常的。
        attention_scores +=adder 

    attention_scores = torch.mul(attention_scores,d_sqrt) #[B,N,F,T]

    attention_porbs = F.softmax(attention_scores) # [B,N,F,T]
    attention_porbs = nn.Dropout(0.9)(attention_porbs)
    
    v = torch.reshape(to_tensor,(batch_size,seq_length,num_attention_heads, attention_head_size)) # [B,T,N,H]
    v = torch.transpose(v, 2,1) # [B,N,T,H]

    y = torch.matmul(attention_porbs, v) #[B,N,F,T] * [B,N,T,H] -> [B,N,F,H]
    y = torch.transpose(y, 2,1) # [B,F,N,H]
    y = torch.reshape(to_tensor,(batch_size,seq_length,num_attention_heads*attention_head_size)) # [B,F,N*H]
    return y


if __name__ == "__main__":
    # 创建mask张量
    attention_mask = create_attention_mask(from_seq_length, to_mask)
    # 把mask张量和embedding传入给attention层
    y = attention_layer(embedding, embedding,attention_mask)
    # 最后返回的张量的shape和输入的shape一样。所以transformer 中使用了for循环，讲上一层attention_layer的输出作为下一层attention_layer的输入
    print(y.shape)

    
