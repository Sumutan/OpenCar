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

"""
输入:左右肩坐标x1,y1,x2,y2
输出:左右肩距
"""

# 设置超参数
learning_rate = 0.0001  # 学习率
net_frame = [1, 3, 1]  # 各层感知机数量
iteration = 70000  # 迭代次数
early_stop=5

logger = SummaryWriter(log_dir="data/log")
# 准备训练数据
# data = readExcel.read_exceldata([0, 1, 2, 3, 4, 5], tablePass='points_train.xlsx', print=True)  # ,从excel读入数据到data
data = readExcel.read_exceldata('A,B,K', tablePass='lengths.xlsx', print=True)

print(data)

train_x = []
train_y = []
validation_x = []
validation_y = []
m = len(data)
# 将读入的数据写入训练集列表train_x与标签列表train_y
# for i in range(m):
#     if i % 3 == 0:  # 每5张中有一张放入验证集
#         validation_x.append([data[i][1], data[i][2], data[i][3], data[i][4]])
#         validation_y.append([data[i][5]])
#         print("已读入验证集" + data[i][0])
#     else:
#         train_x.append([data[i][1], data[i][2], data[i][3], data[i][4]])
#         train_y.append([data[i][5]])
#         print("已读入训练集" + data[i][0])

for i in range(m):
    if i % 5 == 0:  # 每5张中有一张放入验证集
        validation_x.append([data[i][1]])
        validation_y.append([data[i][2]])
        print("已读入验证集" + str(data[i][0]))
    else:
        train_x.append([data[i][1]])
        train_y.append([data[i][2]])
        print("已读入训练集" + str(data[i][0]))

# 张量化
x = torch.tensor(train_x).to(torch.float32)
y = torch.tensor(train_y).to(torch.float32)
vx = torch.tensor(validation_x).to(torch.float32)
vy = torch.tensor(validation_y).to(torch.float32)

print(x)

# #数据送GPU计算
# x =x.to(torch.device(f'cuda:{0}'))
# y =y.to(torch.device(f'cuda:{0}'))
# vx =vx.to(torch.device(f'cuda:{0}'))
# vy =vy.to(torch.device(f'cuda:{0}'))

print("数据准备完毕")
print("数据集x个数:", len(x))
print("验证集vx个数:", len(vx))


# m=10   #样本数:m^2
# x=np.random.rand(2*m).reshape(m,-1)
# # m=x.shape[0]
# train_x=[]
# train_y=[]
# for i in range(m):
#     for j in range(m):
#         train_x.append([x[i][0], x[i][1], x[j][0], x[j][1]])
#         train_y.append([pow(pow((x[i][0]-x[j][0]),2)+pow((x[i][1]-x[j][1]),2),0.5)])
#         # train_y.append([x[i][0]+x[j][0]+x[i][1]+x[j][1]])
#
# x=torch.tensor(train_x).to(torch.float32)
# y=torch.tensor(train_y).to(torch.float32)
# print(len(x))
# # print(y)


# Define model


# 定义网络结构

# 网络类
class Net(nn.Module):
    def __init__(self, net_frame):
        super(Net, self).__init__()
        n_input, n_hidden1, n_output = net_frame

        self.hidden1 = nn.Linear(n_input, n_hidden1)
        self.predict = nn.Linear(n_hidden1, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.predict(out)

        return out


# 生成模型对象
net = Net(net_frame)
# net = Net(net_frame).to(device=torch.device(f'cuda:{0}'))
print(net)
# model = NeuralNetwork().to(device)
# print(model)

# 定义训练方法,损失函数,日志间隔
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # .Adam:adam优化器 ,lr:learning_rate 学习率
loss_func = torch.nn.MSELoss()  # MSELoss损失函数
log_step_interval = 100  # 每100轮训练记录一次loss日志用于后期绘图

# plt.ion()
# plt.show()

# 开始训练
loss_best=0
lose_last=0
best_save = False
for t in range(iteration):

    # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()  # 清空梯度（可以不写）
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新网络

    if loss.item() < early_stop and not best_save:  # 如果loss小于4,提前终止一份防止过拟合
        torch.save(net, 'net/best.pt')
        best_save = True
        loss_best=loss.item()
    if t % log_step_interval == 0:
        # 控制台输出一下
        # print(str(t)+"Loss = %.8f" % loss.data)
        global_iter_num = t
        print("global_step:{}, loss:{:.8}".format(global_iter_num, loss.item()))
        logger.add_scalar("train loss", loss.item(), global_step=global_iter_num)
        lose_last = loss.item()
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#         plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.05)
#         print(str(t))


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