from Toolkit import *
import torch
import torch.nn as nn
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class three_peaks():
    def __init__(self, peak_index, trough1, trough2, value):
        self.peak_index = peak_index
        self.trough1 = trough1
        self.trough2 = trough2
        self.value = value
        self.index = peak_index


def read_one_peak(*, data):
    peak_list = []
    trough_list = []
    index_list = []
    length = data.shape[-1]
    data = data.squeeze(0).squeeze(0)
    trough1 = 0
    trough2 = 0
    peak = 0
    begin = 0
    flag1 = 0
    flag2 = 0
    for i in range(500, length - 500):
        if begin == 1 and data[i] > data[i + 1] and data[i] > data[i - 1]:
            peak_list.append(i)
            flag1 = 1
        if begin == 1 and data[i] < data[i + 1] and data[i] < data[i - 1]:
            trough_list.append(i)
            flag2 = 1
        if peak_list.__len__() == 0 and trough_list.__len__() == 0 and data[i] < data[i + 1] and data[i] < data[i - 1]:
            trough_list.append(i)
            begin = 1
        if peak_list.__len__() == 4:
            peak_list = peak_list[1:]
        if trough_list.__len__() == 5:
            trough_list = trough_list[1:]
        if flag1 + flag2 == 2 and peak_list.__len__() == 3 and trough_list.__len__() == 4:
            flag1 = 0
            flag2 = 0
            # 三峰高度差作为排序标准
            value = data[peak_list[0]] - data[trough_list[0]] + data[peak_list[0]] - data[trough_list[1]] + \
                    data[peak_list[1]] - data[trough_list[1]] + data[peak_list[1]] - data[trough_list[2]] + \
                    data[peak_list[2]] - data[trough_list[2]] + data[peak_list[2]] - data[trough_list[3]] + \
                    50*(2*(data[peak_list[1]] - data[trough_list[2]])/(trough_list[2]-peak_list[1]) - (data[peak_list[2]] - data[trough_list[3]])/(trough_list[3]-peak_list[2]) - (data[peak_list[0]] - data[trough_list[1]])/(trough_list[1]-peak_list[0]))
                    # 10*(data[peak_list[0]] - data[trough_list[1]])/(trough_list[1]-peak_list[0]) + \
                    # 20*(data[peak_list[1]] - data[trough_list[2]])/(trough_list[2]-peak_list[1]) + \
                    # 15*(data[peak_list[2]] - data[trough_list[3]])/(trough_list[3]-peak_list[2])
            # 三峰峰值能量作为排序标准
            # value = data[peak_list[0]] * data[peak_list[0]] + \
            #         data[peak_list[1]] * data[peak_list[1]] + \
            #         data[peak_list[2]] * data[peak_list[2]]
            index_list.append(three_peaks(peak_index=peak_list[1], 
                                          trough1=trough_list[0], 
                                          trough2=trough_list[3],
                                          value=value,))
    index_list = sorted(index_list, key=lambda x: (-x.value, x.peak_index))
    # print(peak_list)
    # print(trough_list)
    return index_list

def get_three_peaks_index(data,p=0.9): # 输入滤波后数据 1 1 n
    out = read_one_peak(data=data)
    gate_input = torch.zeros(1,1,50)
    for i in range(out.__len__()):
        mid_data = F.interpolate(data[:,:,out[i].trough1:out[i].trough2],size=50,mode="linear")
        gate_input = torch.cat([gate_input,mid_data],dim=0)
    gate_input = gate_input[1:,:,:]
    muban = gate_input[0:1,:,:]
    output_list = []
    # for i in range(gate_input.shape[0]):
    #     if C_similarity(gate_input[i][0],muban)>p:
    #         output_list.append(out[i])
    for i in range(10):
        output_list.append(out[i])
    return output_list

# output_list = get_three_peaks_index(Sc2)
# for i in range(output_list.__len__()):
#     print(output_list[i].index)


def get_most_class_peaks_index(data,p=0.9):
    out = read_one_peak(data=data)
    gate_input = torch.zeros(1,1,50)
    for i in range(out.__len__()):
        mid_data = F.interpolate(data[:,:,out[i].trough1:out[i].trough2],size=50,mode="linear")
        gate_input = torch.cat([gate_input,mid_data],dim=0)
    gate_input = gate_input[1:,:,:]
    record = []
    list = []
    for i in range(gate_input.shape[0]):
        list=[]
        if record.__len__()==0:
            list=[]
            list.append(i)
            record.append(list)
        else:
            flag=0
            for j in range(record.__len__()):
                if C_similarity(gate_input[i][0], gate_input[record[j][0]][0])>=p:
                    flag=1
                    list=[]
                    record[j].append(i)
                    break
            if flag==0:
                list=[]
                list.append(i)
                record.append(list)
    print(record.__len__())
    k = 0
    for i in range(record.__len__()):
        if record[i].__len__() > record[k].__len__():
            k = i
    output_list = []
    for i in range(record[k].__len__()):
        output_list.append(out[record[k][i]])
    return output_list


# 根据 三峰能量 做提取
# 假设 三峰能量最大的波段是J峰
def local_max_energy(data):
    # 读取切割的三峰片段
    out = read_one_peak(data=data)
    gate_input = torch.zeros(1,1,50)
    for i in range(out.__len__()):
        mid_data = F.interpolate(data[:,:,out[i].trough1:out[i].trough2],size=50,mode="linear")
        gate_input = torch.cat([gate_input,mid_data],dim=0)
    gate_input = gate_input[1:,:,:]
    length = int(gate_input.shape[0])
    output_list = []
    for i in range(int(length/30)):
        output_list.append(out[i])
    return output_list



# 假设同一人的心跳信号具有周期性，连续取多段三峰信号，期望能得到每个相位点的模板，根据相似性匹配分类
# 只取0.9以上的部分观察进行分类是否拥有同样的相位
def class_continuous(data):
    # 不进行排序
    out = read_one_peak(data=data)
    gate_input = torch.zeros(1,1,50)
    for i in range(out.__len__()):
        mid_data = F.interpolate(data[:,:,out[i].trough1:out[i].trough2],size=50,mode="linear")
        gate_input = torch.cat([gate_input,mid_data],dim=0)
    gate_input = gate_input[1:,:,:]
    # 取连续的
    record = []
    list = []
    max_record = []
    for i in range(10,50):
        list=[]
        list.append(i)
        record.append(list)
    for i in range(gate_input.shape[0]):
        max = 0
        max_index = 0
        for j in range(record.__len__()):
            if C_similarity(gate_input[i][0],gate_input[j][0])>max:
                max = C_similarity(gate_input[i][0],gate_input[j][0])
                max_index = j
        if max > 0.95:
            max_record.append(max)
            record[max_index].append(i)
    # max_record = torch.tensor(max_record)
    # plt.plot(max_record)
    # plt.show()
    k = 0
    for i in range(record.__len__()):
        if record[i].__len__() > record[k].__len__():
            k = i
    output_list = []
    for i in range(record[k].__len__()):
        output_list.append(out[record[k][i]])
    return output_list


# 训练模型部分
class Gate(nn.Module):
    def __init__(self, ):
        super(Gate, self).__init__()
        # 用自注意力做编码
        self.kernel_size = 15
        self.conv = nn.Conv1d(1, 1, self.kernel_size, padding=self.kernel_size // 2)
        # sig softmax  每个心跳大约6个波峰  按10个分类
        self.linear = nn.Linear(50,10)

    def forward(self, output): # 15 1 50
        output = self.conv(output)
        output = self.linear(output) # 15 1 2
        output = torch.sigmoid(output)
        output = torch.softmax(output,dim=2)
        # output = torch.argmax(output,dim=2)
        return output


# 可以使用皮尔逊相关系数
# p = Cov(X,Y) / d(X)d(Y)
def train_Gate(*,data,model,epoch=10,lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    criterion = nn.MSELoss()
    LossRecord = []
    length = data.shape[0]
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        output = model(data) # 1 * n   每个数据代表分的类
        label = torch.argmax(output,  dim=2)
        loss = torch.tensor(0.0,requires_grad=True)
        for i in range(length):
            for j in range(i+1, length):
                if label[i] == label[j]:
                    loss = loss + criterion(data[i][0],data[j][0])
                if label[i] != label[j]:
                    loss = loss + C_similarity(data[i][0],data[j][0])
        # for i in range(length):
        #     for j in range(i+1,length):
        #         loss = loss + (1-C_similarity(data[i][0],data[j][0]))*label[i][0][1] + 0.1*label[i][0][0]
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model

class one_peaks():
    def __init__(self, peak_index, value):
        self.value = value
        self.index = peak_index

def getIndexByGrad(data): # 1 1 10000
    datalength = data.shape[-1]
    length = 300
    reslist = []
    for i in range(length, datalength - length):
        if (data[0][0][i]-data[0][0][i-1]) >= (data[0][0][i-1]-data[0][0][i-2]) and (data[0][0][i]-data[0][0][i-1]) >= (data[0][0][i+1]-data[0][0][i]):
            value = (data[0][0][i]-data[0][0][i-1])
            res = 0
            for j in range(100):
                if(data[0][0][i+j]>data[0][0][i+j-1] and data[0][0][i+j]>data[0][0][i+j+1]):
                    res = j
                    break
            reslist.append(one_peaks(i+res, value))
    reslist = sorted(reslist, key=lambda x: (-x.value, x.index))
    output_list = []
    for i in range(15):
        output_list.append(reslist[i].index)
    return output_list