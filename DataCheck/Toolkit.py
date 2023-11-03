import torch
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
from matplotlib import cm
import torch.nn as nn
import random
from scipy import signal

# BCG滤波器
def bcgFilter(bcg, fs=125):
    Nyquist = fs / 2
    # 高通滤波 1Hz
    b, a = signal.butter(2, 0.6 / Nyquist, 'high')
    bcg = signal.filtfilt(b, a, bcg)

    # 低通滤波 12Hz
    b2, a2 = signal.butter(2, 15.0 / Nyquist)
    bcg = signal.filtfilt(b2, a2, bcg)
    return bcg

def tensor_from_csv(*,PATH,begin,end):
    data = np.array(pd.read_csv(PATH))[begin:end]
    data = data.astype(np.float32)
    data = torch.from_numpy(data)

    return data

# 调用滤波模型 125hz or 1000hz  输入数据形状 x1 1 x2
def modify_data(*,data):#model1,model2
    # 必须使用cuda 模型中部分数据经过cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 读取模型
    import sys
    sys.path.append(r"D:/BCGUnet/PreUnet")
    PATH = 'D:/BCGUnet/newtest/testmodel/0314-125test1.pth'
    model1 = torch.load(PATH)
    PATH = 'D:/BCGUnet/newtest/testmodel/0314-125test2.pth'
    model2 = torch.load(PATH)
    model1 = model1.eval().to(device=device)
    # model1 = model1.to(device=device)
    model2 = model2.eval().to(device=device)
    # model2 = model2.to(device=device)
    print("读取数据滤波模型！")
    data = data.view(1,1,data.shape[0])
    data = data.to(device=device)
    Sc = model1(data)
    Sr = model2(data - Sc)
    data = data.cpu()
    Sc = Sc.cpu()
    Sr = Sr.cpu()

    return data-Sc.detach().numpy()-Sr.detach().numpy()

def trans_to_ones(data):
    max = torch.max(data)
    min = torch.min(data)
    ans = data.clone()
    for i in range(data.shape[-1]):
        ans[i] = (data[i]-min)/(max-min)*20 - 10
    return ans.unsqueeze(0).unsqueeze(0)


def C_similarity(data1,data2):
    ans = torch.inner(data1,data2)/(torch.pow(torch.sum(data1*data1),0.5)*torch.pow(torch.sum(data2*data2),0.5))
    return ans

# 中值滤波降采样  1 1000plt.subplot(211)
def avgpool(*,data,stride):
    length = data.shape[-1]
    ans = torch.zeros(1,1,int(length/stride)-2)
    for i in range(int(length/stride)-2):
        ans[0][0][i] = torch.sum(data[0][0][i*stride:i*stride+stride])/stride
    return ans

# 将数据修正类型
def xiuzhenggeshi(data,begin,end):
    data = data.reshape(1, end - begin)
    data = data.astype(np.float32)
    data = torch.from_numpy(data)
    return data

# 根据血压读取单个心跳
def read_Rebap_peak(data_rebap):
    data_length = data_rebap.shape[-1]
    data_rebap = data_rebap.view(data_length)
    flag = -50
    index = []
    for i in range(5,data_length-1):
        if data_rebap[i]>2.0 and data_rebap[i]>data_rebap[i-1] and data_rebap[i]>data_rebap[i+1] and data_rebap[i]>data_rebap[i-3]:
            if i-flag >= 50:
                # print(i)
                index.append(i)
                flag = i
    index = torch.tensor(index)
    print(index)
    return index

# 中值滤波降采样  1 1000plt.subplot(211)
def avgpool(*,data,stride):
    length = data.shape[-1]
    ans = torch.zeros(1,1,int(length/stride)-2)
    for i in range(int(length/stride)-2):
        ans[0][0][i] = torch.sum(data[0][0][i*stride:i*stride+20])/stride
    return ans

# 将数据修正类型
def xiuzhenggeshi(data,begin,end):
    data = data.reshape(1, end - begin)
    data = data.astype(np.float32)
    data = torch.from_numpy(data)
    return data

# 根据血压读取单个心跳
def read_Rebap_peak(data_rebap):
    data_length = data_rebap.shape[-1]
    data_rebap = data_rebap.view(data_length)
    flag = -50
    index = []
    for i in range(5,data_length-1):
        if data_rebap[i]>2.0 and data_rebap[i]>data_rebap[i-1] and data_rebap[i]>data_rebap[i+1] and data_rebap[i]>data_rebap[i-3]:
            if i-flag >= 50:
                # print(i)
                index.append(i)
                flag = i
    index = torch.tensor(index)
    print(index)
    return index

class grad_class():
    def __init__(self,grad,index):
        self.grad = grad
        self.index = index

# 根据125hz数据波形读取心跳节拍

def read_bcg_peak(data):
    data_length = data.shape[-1]
    data = data.view(data_length)
    ordered_grad_index = []
    trough = 0
    peak1 = 0
    peak2 = 0
    for i in range(10,int(data_length)-300):
        if data[i+1]<data[i] and data[i-1]<data[i]:
            peak1 = peak2
            peak2 = i
            for j in range(peak1,peak2):
                if data[j+1]>data[j] and data[j-1]>data[j]:
                    trough = j
                    # if peak1 < data_length-300 and torch.abs(data[peak1] - data[peak2])<=40:
                    if peak1 < data_length - 300:
                        ordered_grad_index.append(grad_class((data[peak1]-data[trough])+(data[peak2]-data[trough]), peak1)) # 0322
                        # ordered_grad_index.append(grad_class((data[peak1] - data[trough]) , peak1))
                    break

    ordered_grad_index = sorted(ordered_grad_index, key=lambda x:(-x.grad,x.index))
    # for i in range(len(ordered_grad_index)):
    #     print(ordered_grad_index[i].index)
    return ordered_grad_index

# def read_bcg_peak(data):
#     data_length = data.shape[-1]
#     data = data.view(data_length)
#     ordered_grad_index = []
#     peak_length = 100
#     for i in range(int(data_length/peak_length)-1):
#         grad_max = 0
#         grad_min = 0
#         k = data_length
#         for j in range(i*peak_length,(i+1)*peak_length-20):
#             if data[j] - data[j+16] + data[j+30] - data[j+16] > grad_max:
#                 grad_max = data[j] - data[j+16] + data[j+30] - data[j+16]
#                 k = j
#         if k < data_length-200:
#             ordered_grad_index.append(grad_class(grad_max,k))
#     ordered_grad_index = sorted(ordered_grad_index, key=lambda x:(-x.grad,x.index))
#     for i in range(len(ordered_grad_index)):
#         print(ordered_grad_index[i].index)
#     return ordered_grad_index


# 输入数据   n 1 100
def ones_data(data):
    # print(data.shape)
    ans = torch.zeros(data.shape)
    for i in range(data.shape[0]):
        max = torch.max(data[i][0])
        min = torch.min(data[i][0])
        for j in range(data.shape[2]):
            ans[i][0][j] = (data[i][0][j] - min)/(max-min) * 1000 - 500
    return ans

# 生成ResUnet所需数据
def get_ResUnet_data(*,Pathlist,oneperson_begin,oneperson_end):
    # data = torch.zeros(1, 1, 900)
    data = torch.load(Pathlist[0])
    data = torch.zeros(1, 1, data.shape[-1])
    for it in Pathlist:
        data1 = torch.load(it)
        data1 = data1[oneperson_begin:oneperson_end,:,:]
        data = torch.cat([data,data1],dim=0)
    return data[1:,:,:]

def showData(*,data,persons):
    datalength = data.shape[0]
    each_nums = int(datalength//persons)
    for i in range(persons):
        plt.subplot(persons,1,i+1)
        for j in range(each_nums):
            plt.plot(data[i*each_nums+j][0].detach().numpy())
    plt.show()