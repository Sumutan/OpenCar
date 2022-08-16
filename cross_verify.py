"""
该代码用于通过k折交叉验证测试模型的精度
将数据集（lenths.xlsx）读入后进行模型训练并评估
trainModel=False为不需要重新训练，直接从net/last.pt读取模型
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

# 设置超参数
learning_rate = 0.0001  # 学习率
net_frame = [7, 3, 6]  # 各层感知机数量
iteration = 100000  # 迭代次数
trainModel = False


class Net(nn.Module):
    def __init__(self, net_frame):
        super(Net, self).__init__()
        n_input, n_hidden1, n_output = net_frame
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.Linear(n_input, n_hidden1),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden1, n_output)
                                 )

    def forward(self, input):
        return self.net(input)


# 输入模型与数据集，验证准确度
def val_acc(net, val_x, val_y):  # 将所有验证集代入模型验证精度，输出各项平均精度
    y = net(val_x)
    print(y)
    error_rate = abs((y - val_y) / val_y)
    print(type(error_rate))
    print("各项误差:\n", error_rate)

    ave_error_rate = torch.mean(error_rate, dim=0)
    print("平均误差：", ave_error_rate)
    return ave_error_rate


# 准备训练数据
data = readExcel.read_exceldata('A,B,C,D,E,F,G,H,I,J,K,L,M,N', tablePass='lengths.xlsx', print=True)

print(data)

train_x = []
train_y = []
validation_x = []
validation_y = []
m = len(data)
# 将读入的数据写入训练集列表train_x与标签列表train_y

cross_acc = []
for c in range(5):
    print("数据总数：", str(m))
    print("第%d折交叉验证" % c)
    for i in range(m):
        if i % 5 == 0:  # 每5张中有一张放入验证集
            validation_x.append([data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7]])
            validation_y.append([data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13]])
            # print("已读入验证集" + str(data[i][0]))
        else:
            train_x.append([data[i][1], data[i][2], data[i][3], data[i][4], data[i][5], data[i][6], data[i][7]])
            train_y.append([data[i][8], data[i][9], data[i][10], data[i][11], data[i][12], data[i][13]])
            # print("已读入训练集" + str(data[i][0]))

    # 张量化
    x = torch.tensor(train_x).to(torch.float32)
    y = torch.tensor(train_y).to(torch.float32)
    vx = torch.tensor(validation_x).to(torch.float32)
    vy = torch.tensor(validation_y).to(torch.float32)

    # print(x)
    # print("数据准备完毕")
    # print("数据集x个数:", len(x))
    # print("验证集vx个数:", len(vx))

    if trainModel:
        net = Net(net_frame)  # 生成模型对象
        # print(net)

        # 定义训练方法,损失函数,日志间隔
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # .Adam:adam优化器 ,lr:learning_rate 学习率
        loss_func = torch.nn.MSELoss()  # MSELoss损失函数
        log_step_interval = 1000  # 每100轮训练记录一次loss日志用于后期绘图

        # 开始训练
        loss_best = 10000
        lose_last = 0
        best_save = False  # 保存最优模型
        for t in range(iteration):
            # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
            prediction = net(x)
            loss = loss_func(prediction, y)
            optimizer.zero_grad()  # 清空梯度（可以不写）
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新网络

            if t % log_step_interval == 0:
                # 控制台输出一下
                # print(str(t)+"Loss = %.8f" % loss.data)
                global_iter_num = t
                print("global_step:{}, loss:{:.8}".format(global_iter_num, loss.item()))
                lose_last = loss.item()
                if lose_last < loss_best:
                    loss_best = lose_last
        print("训练完成")

    else:
        net = torch.load('net/last.pt')
    loss_acc = 0
    lossrate_acc = 0

    # print(vx[i])
    # print("input:", str(vx[i]))
    # prediction = net(vx[i])  # prediction=[a,b,...,f]
    # print("output:", str(prediction))
    # print("label:", str(vy[i]))
    # loss = abs(prediction - vy[i])
    # loss_rate = loss / vy[i]
    # ave_loss_rate = torch.mean(loss_rate, dim=0)
    # print(ave_loss_rate)
    # cross_acc.append(ave_loss_rate)

print(cross_acc)
