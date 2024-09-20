import torch
from torch.autograd import Variable
from torch.autograd import Function

# class Myop1(Function):
#     @staticmethod
#     def forward(self, input):
#         self.save_for_backward(input)
#         output = input.clamp(min=0)
#         return output
#     @staticmethod
#     def backward(self, grad_output):
#         ##
#         input = self.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input<0] = 0
#         return grad_input
# input_1 = Variable(torch.randn(1), requires_grad=True)
# print(input_1)
# relu2 = Myop1.apply
# output_=  relu2(input_1)
# print(output_.grad_fn)

# def relu(input):
#     return Myop1.apply(input)
'''
自定义激活函数
def hardswish_func(input):
    if input<-3:
        return 0
    elif input<3:
        output=(input^2)/6+input/2
        return output
    else:
        return input
def hardswaish_daoshu(input):
    if input<-3:
        return 0
    elif input<3:
        output=input/3+1/2
        return output
    else:
        return 1
class hardswish(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        output=hardswish_func(input)
        return output
    @staticmethod
    def backward(ctx,grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = hardswaish_daoshu(grad_output)
        return grad_input
input_1 = Variable(torch.randn(1), requires_grad=True)
print(input_1)
relu2 = hardswish.apply
output_=  relu2(input_1)
print(output_.grad_fn)
'''

import torch
import numpy as np
data=[[1,2],[2,3]]
a=torch.tensor(data)
np_array=np.array(data)
x_np=torch.from_numpy(np_array)
# print(x_np)
x_ones=torch.ones_like(a)
# print(x_ones)
rand_tensor=torch.rand(2,2)
# print(rand_tensor)
ones_tensor=torch.ones(2,2)
zeros_tensor=torch.zeros(2,2)
# print(ones_tensor)
# print(ones_tensor.device)
# if torch.cuda.is_available():
#     ones_tensor=ones_tensor.to('cuda')
#     print(ones_tensor.device)
# two=torch.ones(2,2)
# two[:,0]=2
# print(two)
# cat=torch.cat([two,two],dim=1) #dim是拼接的维度，
# print(cat)
# print(two.mul(two))#逐个元素相乘
# print(two*two)
# print(two.matmul(two.T))#矩阵乘法
# print(two@two.T)
# two.add_(2)
# print(two)
# num=two.numpy()
# print(num)
# n=np.ones(2)
# # tensor=torch.from_numpy(n)#numpy数组转化为张量
# # print(tensor)
# from torch import nn
# import math
# class mnist(nn.Module):
#     def __init__(self):
#         self.conv1=nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=2,padding=1)
#         self.conv2=nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,stride=2,padding=1)
#         self.conv3=nn.Conv2d(in_channels=16,out_channels=19,kernel_size=3,stride=2,padding=1)
#     def forward(self,x):
#         x=x.view(-1,1,28,28)
#         x=torch.relu(self.conv1(x))
#         x=torch.relu(self.conv2(x))
#         x=torch.relu(self.conv3(x))
#         x=torch.nn.AvgPool2d(x,kernel_size=4)
#         return x.view(-1,x.size(1))
import flask
import app
# def user(username='klkl'):
#     print(username.title())
# def make_shirt(size,brand):
#     print('this T-shirt\'s size is '+str(size)+',and brand is '+brand)
# make_shirt(13,'nike')
# def city_describe(name,country='China'):
#     print(name +' is belong to '+country)
# city_describe('newYork','America')
# def build_person(name,age=''):
#     person={'name':name,'age':age}
#     if age:
#         person['age']=age
#     return person
# player=build_person('Jordan',60)
# print(player)
# def greet_users(names):
#     for name in names:
#         msg="hello,"+name.title()+"!"
#         print(msg)
# users=['dsc','fds','sdsd']
# greet_users(users)
# class Dog():
#     def __init__(self,name,age):
#         self.name=name
#         self.age=age
#     def dog_information(self):
#         print('this dog name is: '+self.name)
#         print('this dog age is:'+str(self.age))
# huang=Dog('xiaohei',age=2)
# huang.dog_information()
# class zangao(Dog):
#     def __init__(self,name,age):
#         super(zangao, self).__init__(name,age)
# zao=zangao(name='ddd',age=3)
# zao.dog_information()

'''
# 自定义激活函数示例
import torch
from torch.autograd import Function
from torch.autograd import gradcheck
class LinearFunction(Function):
    @staticmethod
    # 第一个是ctx，第二个是input，其他是可选参数。
    # 自己定义的Function中的forward()方法，所有的Variable参数将会转成tensor！因此这里的input也是tensor．在传入forward前，autograd engine会自动将Variable unpack成Tensor。
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)  # 将Tensor转变为Variable保存到ctx中
        output = input.mm(weight.t())  # .t()是转置的意思
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)  # unsqueeze(0) 扩展出第0维(原本bias只有1维)
            # expand_as(tensor)等价于expand(tensor.size()), 将原tensor按照新的size进行扩展
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        # 分别代表输入,权值,偏置三者的梯度
        # 判断三者对应的tensor是否需要进行反向求导计算梯度
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)  # 复合函数求导，链式法则
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)  # 复合函数求导，链式法则
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)  # 变回标量
        # sum(0)是指在维度0上面求和;squeeze(0)是指压缩维度0
        # 梯度的顺序和 forward 形参的顺序要对应
        return grad_input, grad_weight, grad_bias

linear = LinearFunction.apply

# 可以通过gradcheck 来检查定义的梯度计算是否正确
input = torch.randn(20, 20, requires_grad=True).double()
weight = torch.randn(20, 20, requires_grad=True).double()
bias = torch.randn(20, requires_grad=True).double()
test = gradcheck(LinearFunction.apply, (input, weight, bias), eps=1e-6, atol=1e-4)
print(test)  # 没问题的话输出True
'''

'''
自定义激活函数示例1
import torch
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs.
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Create random Tensors for weights.
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # To apply our Function, we use Function.apply method. We alias this as 'relu'.
    relu = MyReLU.apply

    # Forward pass: compute predicted y using operations; we compute
    # ReLU using our custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
'''

'''
import torch.nn as nn
class SelfDefinedHardSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp):
        if inp <= -3:
            result = 0
        elif inp < 3:
            result = (1/6)*inp.pow(2)+1/2*inp
        else:
            result = inp
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        # ctx.saved_tensors is tuple (tensors, grad_fn)
        result, = ctx.saved_tensors
        if result <= -3:
            return 0
        elif result < 3:
            return 1/3*result+1/2
        else:
            return 1

class HardSwish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = SelfDefinedHardSwish.apply(x)
        return out
'''

'''
# 自定义激活函数示例2
from torch.autograd import Function
import torch

class MyReLU(Function):

    @staticmethod
    def forward(ctx, input_):
        # 在forward中，需要定义MyReLU这个运算的forward计算过程
        # 同时可以保存任何在后向传播中需要使用的变量值
        ctx.save_for_backward(input_)  # 将输入保存起来，在backward时使用
        output = input_.clamp(min=0)  # relu就是截断负数，让所有负数等于0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
        # dloss / doutput就是输入的参数grad_output、
        # 因此只需求relu的导数，在乘以grad_output
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input_ < 0] = 0  # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
        return grad_input

x = torch.rand(4, 3, 5, 5)
myrelu = MyReLU.apply  # Use it by calling the apply method:
output = myrelu(x)
print(output.shape)
'''

'''
# 自定义激活函数示例3
# 在前行传播函数中使用if这种方法不可行，无法进行比较。
import torch
from torch.autograd import Function
class HardSwish(Function):
    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        if input <= -3:
            output = 0
        elif input < 3:
            output = 1/6*input.pow(2)+1/2*input
        else:
            output = input
        ctx.save_for_backward(output)
        return output
    @staticmethod
    def backward(ctx,grad_output):
        input,output=ctx.saved_tensors
        grad_input=grad_output*output
        grad_input[input <= -3] = 0
        grad_input[input > -3 & input < 3] = 1/3*input+1/2
        grad_input[input >=3] = 1
        return grad_input

x = torch.rand(4, 3, 5, 5)
myrelu = HardSwish.apply  # Use it by calling the apply method:
output = myrelu(x)
print(output.shape)
'''