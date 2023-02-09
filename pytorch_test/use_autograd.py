import torch , torchvision

model =  torchvision.models.resnet18(pretrained = True)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

prediction = model(data) #forward pass

loss = (prediction-labels).sum()
loss.backward() # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9) #学习率为 0.01，动量为 0.9

optim.step() #调用.step()启动梯度下降。 优化器通过.grad中存储的梯度来调整每个参数


