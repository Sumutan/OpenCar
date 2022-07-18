"""
输入图片进行一次推理
未完成
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import netron
# Create data loaders.
# train_dataloader = DataLoader(training_data, batch_size=batch_size)
# test_dataloader = DataLoader(test_data, batch_size=batch_size)
from tensorboardX import SummaryWriter
import readExcel
import time

print("last.pt")
# 模型效果测试(训练集)
loss_acc = 0
lossrate_acc = 0
for i in range(len(x)):
    # print("测试数据",i)
    print("input:", str(x[i]), end="")
    prediction = net(x[i])
    loss = abs(prediction - y[i])
    print("肩宽:" + str(prediction.data.numpy()) + " 实际肩宽:" + str(y[i]))
    lossrate = abs(prediction.data.numpy() - y[i].numpy()) / y[i].numpy()
    loss_acc += loss.data.numpy()
    lossrate_acc += lossrate
loss_acc /= len(x)
lossrate_acc /= len(x)
print("last.pt训练集平均误差:", loss_acc)
print("last.pt训练集平均误差率:", lossrate_acc)

# 模型效果测试(测试集)net
loss_acc = 0
lossrate_acc = 0
for i in range(len(vx)):
    # print("测试数据",i)
    print("input:", str(vx[i]), end="")
    prediction = net(vx[i])
    loss = abs(prediction - vy[i])
    print("身高:" + str(prediction.data.numpy()) + " 实际身高:" + str(vy[i]))
    lossrate = abs(prediction.data.numpy() - vy[i].numpy()) / vy[i].numpy()
    loss_acc += loss.data.numpy()
    lossrate_acc += lossrate
loss_acc /= len(vx)
lossrate_acc /= len(vx)
print("last.pt测试集平均误差:", loss_acc)
print("last.pt测试集平均误差率:", lossrate_acc)
print("lose_last:",str(lose_last))
# 保存模型
torch.save(net, 'net/last.pt')


"""接下来用之前提前中断保存的best.pt测试"""
print()
print("best.pt")
# 模型效果测试(训练集)
net = Net(net_frame)
loss_acc = 0
lossrate_acc = 0
for i in range(len(x)):
    # print("测试数据",i)
    print("input:", str(x[i]), end="")
    prediction = net(x[i])
    loss = abs(prediction - y[i])
    print("身高:" + str(prediction.data.numpy()) + " 实际身高:" + str(y[i]))
    lossrate = abs(prediction.data.numpy() - y[i].numpy()) / y[i].numpy()
    loss_acc += loss.data.numpy()
    lossrate_acc += lossrate
loss_acc /= len(x)
lossrate_acc /= len(x)
print("best.pt训练集平均误差:", loss_acc)
print("best.pt训练集平均误差率:", lossrate_acc)

# 模型效果测试(测试集)
loss_acc = 0
lossrate_acc = 0
for i in range(len(vx)):
    # print("测试数据",i)
    print("input:", str(vx[i]), end="")
    prediction = net(vx[i])
    loss = abs(prediction - vy[i])
    print("肩宽:" + str(prediction.data.numpy()) + " 实际肩宽:" + str(vy[i]))
    lossrate = abs(prediction.data.numpy() - vy[i].numpy()) / vy[i].numpy()
    loss_acc += loss.data.numpy()
    lossrate_acc += lossrate
loss_acc /= len(vx)
lossrate_acc /= len(vx)
print("best.pt测试集平均误差:", loss_acc)
print("best.pt测试集平均误差率:", lossrate_acc)
print("lose_bast:",str(loss_best))