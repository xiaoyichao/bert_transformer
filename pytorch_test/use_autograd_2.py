'''
Author: root root@haohaozhu.com
Date: 2023-02-09 16:42:08
LastEditors: root root@haohaozhu.com
LastEditTime: 2023-02-09 16:51:19
FilePath: /bert_transformer/pytorch_test/use_autograd_2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch , torchvision


a = torch.tensor([2.,3.],requires_grad=True)
b = torch.tensor([6.,4.],requires_grad=True)

Q= 3*a**3 -b**2 #假设Q是神经网络的误差

external_grad = torch.tensor([1.,1.])
Q.backward(gradient=external_grad)

print(9*a**2==a.grad)
print(-2*b == b.grad)

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")




