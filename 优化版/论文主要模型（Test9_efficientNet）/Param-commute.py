# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# import torch
# import numpy as np

# tensor=torch.randn(2,2)
# # print(tensor)
# tensor.resize(1,4)
# print(tensor)
# print(tensor[0])
# print(tensor[:,0])
# print(tensor.size())
# print(tensor[1,-1])
# b=tensor**2
# c=tensor*3
# print(b);print(c)
# a=torch.Tensor(2,2)
# b=torch.tensor(5)
# print(a);print(b)
# a=np.ones((2,2),dtype=np.float32)
# print(a)
# b=torch.from_numpy(a) # numpy格式转化为torch
# print(b)
# c=torch.Tensor(a) # numpy直接传入Tensor
# print(c)

# import torchvision
# a=torch.tensor([[1,1,1],
#                 [2,2,2],
#                 [3,3,3]],dtype=torch.float32)
# print(a);print(a.size())
# print(a.flatten())
# b=torch.tensor([[5,5,5],
#                 [6,6,6],
#                 [7,7,7]],dtype=torch.float32)
# c=torch.stack([a,b])
# print(c);print(c.size())

# a=torch.rand(2,2)
# print(a)
# b=torch.tensor([4,3])
# print(a+b) # 维数不同的张量相加时自动广播为相同维度
# c=torch.tensor([[1],
#                [2]])
# print(a+c)
# a=torch.tensor([[1,2,3],
#                 [3,4,2],
#                 [5,7,1]])
# print(a.sum())
# print(a.sum(dim=0)) # tensor([ 9, 13,  6])
# print(a.sum(dim=1)) # tensor([ 6,  9, 13])
# print(a.max())
# print(a.max(dim=0))
#     # values=tensor([5, 7, 3]),
#     # indices=tensor([2, 2, 0]))
# print(a.max(dim=1))
#     # values=tensor([3, 4, 7]),
#     # indices=tensor([2, 1, 1]))

# import torch
# from matplotlib import pyplot as plt
# from IPython import display
# import numpy as np
# import math
# x=np.linspace(0,0.1,2)
# y=x
# plt.plot(x,y)
# it=iter([1,2,3])
# while True:
#     try:
#         x=next(it)
#         print(x)
#     except StopIteration:
#         break
# print(torch.__version__)
# a=torch.tensor([[1,3,4],
#                 [3,5,2]])
# scatter=a[0,0] # 直接使用索引得到的依然是张量
# print(scatter)
# print(scatter.item()) # item()将张量转化为标量。

# ten=torch.tensor(([2]))
# scalar=ten.item()
# print('tensor=', ten, 'scalar=', scalar)
# x=torch.rand(2,2)
# y=torch.rand(2,2)
# print('x=',x,'y=',y)
# device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# x=x.to(device)
# y=y.to(device)
# z=x+y
# print('z=',z)

# x=torch.rand(2,2,requires_grad=True)
# y=x.sum()
# print(x)
# print(y)
# print(y.grad_fn)
# print(y.backward())
# print(x.grad)
#
# y.backward()
# print(x.grad)
#
# y.backward()
# print(x.grad)
#
# x.grad.data.zero_()
# print(x.grad)
#
# y.backward()
# print(x.grad)

# import torch
# from matplotlib import pyplot as plt
# from IPython import display
# import numpy as np
# import math

# data=np.array(([1,3,4]))
# print(torch.tensor(data))
# print(torch.Tensor(data))
# print(torch.from_numpy(data))

# def flatten(t):
#     t=t.reshape(1,-1)
#     t=t.squeeze()
#     return t
# t=torch.ones(2,2)
# print(flatten(t))

# class tiger:
#     def __init__(self,name):
#         self.name=name
#     def set_name(self,name):
#         self.name=name
#     def print_name(self):
#         print(self.name)
# tiger1=tiger('jm')
# tiger1.print_name()
# tiger1.set_name('gd')
# tiger1.print_name()

# import torch
# import torch.nn as nn
# class network(nn.Module):
#     def __init__(self):
#         super(network,self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
#         self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
# # 线性层，也叫全连接层，12*4*4的作用是将高维张量展开成秩为1的张量，然后再进入神经网络处理
# # 但是为什么是4*4？
# # 经过卷积和池化等操作最后将输入特征变成了4*4大小的特征
#         self.fc2 = nn.Linear(in_features=120, out_features=60)
#         self.out = nn.Linear(in_features=60, out_features=10)
#     def forward(self,t):
#         return t
# obj=network()
# print(obj)
# print(obj.conv1.weight.shape)
# # 第一个卷积层的shape
# # 有四个轴，6表示6个卷积核，1表示单通道，5,5表示卷积核的尺寸
# print(obj.conv2.weight.shape)
# # 12个5*5滤波器，有6个输入通道来自于上一层
# import torch
# t1 = torch.tensor([1,1,1])
# t2 = torch.tensor([2,2,2])
# t3 = torch.tensor([3,3,3])
# print(torch.stack((t1,t2,t3),dim=0))
# print(torch.stack((t1,t2,t3),dim=1))

# import torch
# a=torch.tensor([[1,2,3],
#                 [2,4,5]],dtype=float)
# print(a.matmul(a.T)) # 张量和其转置的乘积

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1 input image channel, 6 output channels, 5x5 square convolution
#         # kernel
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         # an affine operation: y = Wx + b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         # If the size is a square, you can specify with a single number
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# net = Net()
# print(net)
# input = torch.randn(1, 1, 32, 32)
# out = net(input)
# print(out)

# import matplotlib.pyplot as plt
# x=[0,1,2,3,4,5]
# y=[0,1,4,9,16,25] # 当只给定y数据时，x默认从0开始。
# plt.plot(y,linewidth=2)
# plt.title("square numbers",fontsize=24)
# plt.xlabel("value",fontsize=14)
# plt.ylabel("square",fontsize=14)
# plt.tick_params(axis='both',labelsize=14)
# plt.show()

# import torch
# from torch import nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# net = nn.Sequential(nn.Linear(20, 256),nn.ReLU(),nn.Linear(256, 10))
# x = torch.randn(2,20)
# print(net(x))
# class MLP(nn.Module):
#     def __init__(self,**kwargs):
#         super(MLP,self).__init__(**kwargs)
#         self.hidden=nn.Sequential(nn.Linear(20,256),nn.ReLU())
#         self.output=nn.Linear(256,10)
#     def forward(self,x):
#         return self.output(self.hidden(x))

# from efficientnet_pytorch import EfficientNet
# import torch
# import torchvision
# # model = EfficientNet.from_pretrained('efficientnet-b0')
# resnet50=torchvision.models.resnet50(pretrained=True)
# # 参数量计算方法1
# # total = sum(p.numel() for p in resnet50.parameters())
# # print(total)
# # print("Total params: %.2fM" % (total/1e6))
#
# # 参数量和计算量计算方法2，利用thop模块，未安装则使用pip install thop安装
# from thop import profile
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(resnet50, inputs=(input,))
# print(flops)
# print(params)

# # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # 论文中swish和hard swish激活函数图像的绘制
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import pandas as pd
# import heapq
# from scipy import signal
#
# # x = np.linspace(-10, 10, 100)   #分别代表最小，最大，数量， 生成一个等差数列
# # y = [math.sin(t) for t in x]
# # swish函数的图像绘制
# z=np.linspace(-7,7,100)
# swish=[h/(1+math.exp(-h)) for h in z]
# plt.plot(z, swish,'r',label='Swish',linestyle='--')
#
# # hardswish函数的图像绘制
# hardswish=np.linspace(-7,7,100)
# interval0 = [1 if (i<=-3) else 0 for i in hardswish]
# interval1 = [1 if (i>-3 and i<=3) else 0 for i in hardswish]
# interval2 = [1 if (i>3) else 0 for i in hardswish]
# table=hardswish*0*interval0+(hardswish*(hardswish+3)*interval1)/6+(hardswish*interval2)
# plt.plot(hardswish,table,'g',label='Hard Swish')
#
# plt.xlabel('x',fontsize=18)
# plt.ylabel('f (x)',fontsize=18,fontproperties='times new roman')
#
# # #ReLU激活函数绘制
# # relu=np.linspace(-7,7,100)
# # relu_interval1=[1 if (i<=0) else 0 for i in relu]
# # relu_interval2=[1 if (i>0) else 0 for i in relu]
# # relu_plot=relu*0*relu_interval1+relu*relu_interval2
# # plt.plot(relu,relu_plot,'k',label='ReLU')
#
# # plt.title('swish and hardswish')
# # plt.legend(['swish','hardswish'])
#
# plt.rcParams.update({'font.size': 16})# 改变标签的字体大小
# # ax = plt.subplot(111)
# # ax.set_xlabel(..., fontsize=18)
# # ax.set_ylabel(..., fontsize=18)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
#
# #限制x、y坐标的刻度范围，x显示范围(-7,8)，间距为2
# # y显示刻度范围(-1,8)，间距为1。
# x_area=range(-7,8,2)
# y_area=range(-1,8,1)
# plt.xticks(x_area)
# plt.yticks(y_area)
# plt.legend(loc='upper left',frameon=False)
# plt.show()
# # %%%%%%%%%%%%%%%%%%%%%

# #%%%%%%%%%%%%%%%%
# # 计算模型参数量和计算量，使用thop模块
# import math
# import torchvision
# model=torchvision.models.vgg16(pretrained=True)
# # print(model)
# import torch
# print(torch.cuda.is_available())
# import thop
# from thop import profile
# dummy_input = torch.randn(1, 3, 224, 224)
# flops, params = profile(model, (dummy_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
# # 输入input的第一维度是批量(batch size)，批量的大小不回影响参数量， 计算量是batch_size=1的倍数
# # profile(net, (inputs,))的 (inputs,)中必须加上逗号，否者会报错
# # %%%%%%%%%%%%%%%%%%%%%%%%%

# # %%%%%%%%%%%%%%%%
# # 逐渐降低学习率
# import torch
# import math
# import torchvision
# net=torchvision.models.vgg16(pretrained=True)
# def rule(epoch):
#     lamda = math.pow(0.5, int(epoch / 10))
#     return lamda
# optimizer = torch.optim.SGD(net.parameters(),lr=0.1)
# # optimizer = torch.optim.SGD([{'params': net.parameters(), 'initial_lr': 0.5}], lr = 0.5)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = rule)
# scheduler1=torch.optim.lr_scheduler.StepLR(optimizer,step_size=4,gamma=0.1)
# for i in range(20):
#     print("lr of epoch", i, "=>", optimizer.param_groups[0]['lr'])
#     optimizer.step()
#     scheduler1.step()
# # %%%%%%%%%%%%%%%%%%%%%

# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import StepLR
# import itertools
# initial_lr = 0.1
# class model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3)
#
#     def forward(self, x):
#         pass
# net_1 = model()
# optimizer_1 = torch.optim.Adam(net_1.parameters(), lr=initial_lr)
# scheduler_1 = StepLR(optimizer_1, step_size=3, gamma=0.1)
#
# print("初始化的学习率：", optimizer_1.defaults['lr'])
#
# for epoch in range(1, 11):
#     # train
#
#     optimizer_1.zero_grad()
#     optimizer_1.step()
#     print("第%d个epoch的学习率：%f" % (epoch, optimizer_1.param_groups[0]['lr']))
#     scheduler_1.step()
# import torch
# print(torch.__version__)

#%%%%%%%%%%%%%%%%
# 计算模型参数量和计算量，使用thop模块
import math
import torchvision
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 注意之前的模型使用的是gpu进行训练，得到的也是gpu类型的模型
# 所以后边输入的随机张量也要转移到gpu上才行。
model=torch.load('E:/PaperModels/MobileNetV2/MobileNetV2_Model/MobileNetV2-10-all.pth')
import thop
model.to(device)
from thop import profile
dummy_input = torch.randn(28, 3, 224, 224,device=device)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

# 输入input的第一维度是批量(batch size)，批量的大小不回影响参数量， 计算量是batch_size=1的倍数
# profile(net, (inputs,))的 (inputs,)中必须加上逗号，否者会报错
# %%%%%%%%%%%%%%%%%%%%%%%%%
