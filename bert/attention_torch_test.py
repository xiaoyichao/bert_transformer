import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, batch_size, seq_len, dim, head_num):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim
        self.head_num = head_num
        self.head_size = self.dim//self.head_num
        self.dk = math.sqrt(self.head_size)
        self.q_layer = nn.Linear(self.dim, self.dim)
        self.k_layer = nn.Linear(self.dim, self.dim)
        self.v_layer = nn.Linear(self.dim, self.dim)

    def forward(self, from_tensor, to_tensor):
        q = self.q_layer(from_tensor)
        k = self.k_layer(to_tensor)
        v = self.v_layer(to_tensor)

        q = self.transpose(q)
        k = self.transpose(k)
        v = self.transpose(v)

        atten_scores = torch.matmul(q, k.transpose(-1, -2))/self.dk
        atten_scores = F.softmax(atten_scores, dim=-1)
        atten_probs = torch.matmul(atten_scores, v)
        atten_probs = atten_probs.transpose(1, 2).contiguous()
        output = atten_probs.view(self.batch_size, self.seq_len,
                                 self.head_num*self.head_size)
        return output

    def transpose(self, input_tensor):
        input_tensor = input_tensor.view(
            self.batch_size, self.seq_len, self.head_num, self.head_size)
        output_tensor = input_tensor.transpose(1, 2)
        return output_tensor


if __name__ == "__main__":
    batch_size, seq_len, dim, head_num = 32, 512, 768, 12
    embeddings = torch.rand(batch_size, seq_len, dim)
    attention = Attention(batch_size, seq_len, dim, head_num)
    y = attention.forward(embeddings, embeddings)
    print(y.shape)
