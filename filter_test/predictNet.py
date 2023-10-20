from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
import numpy as np
# from Toolkit import *
import torch.nn.functional as F

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

class predictNet(nn.Module):
    def __init__(self, kernel_size1, kernel_size2):
        super().__init__()
        self.kernel_size1 = kernel_size1
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size1,
            padding=self.kernel_size1 // 2,
            padding_mode='reflect',
        )
        self.kernel_size2 = kernel_size2
        self.conv2 = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size2,
            padding=self.kernel_size2 // 2,
            padding_mode='reflect',
        )
        self.avgpool1 = nn.AvgPool1d(kernel_size=kernel_size1, stride=1, padding=kernel_size1//2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel_size2, stride=1, padding=kernel_size2//2)
        # self.linear1 = nn.Linear(kernel_size1+1, 1)
        # self.linear2 = nn.Linear(kernel_size2+1, 1)
        pred_length1 = 3
        pred_length2 = 3
        self.linear1 = nn.Linear(kernel_size1+1, pred_length1)
        self.linear2 = nn.Linear(kernel_size2+1, pred_length2)
        # self.linear1 = nn.Sequential(
        #     nn.Linear(kernel_size1+1, 4*(kernel_size1+1), bias=True),
        #     nn.ReLU(0.2),
        #     nn.Linear(4*(kernel_size1+1),1, bias=True)
        # )
        # self.linear2 = nn.Sequential(
        #     nn.Linear(kernel_size2+1, 4*(kernel_size2+1), bias=True),
        #     nn.ReLU(0.2),
        #     nn.Linear(4*(kernel_size2+1),1, bias=True)
        # )
        self.avgpool3 = nn.AvgPool1d(kernel_size=pred_length1, stride=pred_length1)
        self.avgpool4 = nn.AvgPool1d(kernel_size=pred_length2, stride=pred_length2)
        
    def forward(self, output):
        # input shape : N 1 10000
        # # 卷积
        # output1 = self.conv1(input).squeeze(1)
        # output2 = self.conv2(input).squeeze(1)
        # 线性层
        output = torch.nn.functional.normalize(output.squeeze(1), dim=1)
        output1 = output.squeeze(1)
        output2 = output.squeeze(1)
        avg1 = self.avgpool1(output1)
        avg2 = self.avgpool1(output2)
        # 移位拼接
        output1 = self.DIYroll(output1,self.kernel_size1)
        output2 = self.DIYroll(output2,self.kernel_size2)
        # 添加辅助趋势项
        output1 = torch.cat([output1, avg1],dim=0)
        output2 = torch.cat([output2, avg2],dim=0)
        # 线性变换 预测学习
        output1 = self.linear1(output1.t()).t()
        output2 = self.linear2(output2.t()).t()
        # 取预测分段的均值
        output1 = torch.flatten(output1.t()).unsqueeze(0)
        output2 = torch.flatten(output2.t()).unsqueeze(0)
        
        output1 = self.avgpool3(output1)
        output2 = self.avgpool4(output2)# 1 1 5600
        output1 = torch.nn.functional.normalize(output1, dim=1)
        output2 = torch.nn.functional.normalize(output2, dim=1)
        return (output1 + output2).unsqueeze(1)
    
        
    def DIYroll(self, input, shift):
        output = input
        for i in range(1,shift):
            output = torch.cat([output, 
                                torch.roll(input,shifts=i,dims=1),
                                ],dim=0)
        return output
    
    def returnAns(self, output):
        # input shape : N 1 10000
        # # 卷积
        # output1 = self.conv1(input).squeeze(1)
        # output2 = self.conv2(input).squeeze(1)
        # 线性层
        output = torch.nn.functional.normalize(output.squeeze(1), dim=1)
        output1 = output.squeeze(1)
        output2 = output.squeeze(1)
        avg1 = self.avgpool1(output1)
        avg2 = self.avgpool1(output2)
        # 移位拼接
        output1 = self.DIYroll(output1,self.kernel_size1)
        output2 = self.DIYroll(output2,self.kernel_size2)
        # 添加辅助趋势项
        output1 = torch.cat([output1, avg1],dim=0)
        output2 = torch.cat([output2, avg2],dim=0)
        # 线性变换 预测学习
        output1 = self.linear1(output1.t()).t()
        output2 = self.linear2(output2.t()).t()
        # 取预测分段的均值
        output1 = torch.flatten(output1.t()).unsqueeze(0)
        output2 = torch.flatten(output2.t()).unsqueeze(0)
        output1 = self.avgpool3(output1)
        output2 = self.avgpool4(output2)# 1 1 5600
        output1 = torch.nn.functional.normalize(output1, dim=1)
        output2 = torch.nn.functional.normalize(output2, dim=1)
        return output1.unsqueeze(1), output2.unsqueeze(1)
    
    

def trainPredictNet(*,model,data,T,lr=0.001,epoch=10):
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1.0)
    criterion = nn.MSELoss()
    LossRecord = []
    data = data.data
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        output1 = model(data)
        loss = criterion(data+T, output1+T) * 1 + torch.sum(output1[0][0] * output1[0][0]) * 0.1 + torch.sum(output1[0][0]) * torch.sum(output1[0][0]) * 0.1
        # loss = criterion(data, output1) * 1 \
        #        + ( torch.sum(torch.pow(model.conv1.weight, 2)) + torch.sum(torch.pow(model.conv2.weight, 2)) ) * 1 \
        #        + torch.sum(torch.abs(output1[:, :, 1:] - output1[:, :, :output1.shape[2] - 1])) * 1 \
        #        + torch.sum(output1[0][0] * output1[0][0]) * 0.1 \
        #        + torch.sum(output1[0][0]) * torch.sum(output1[0][0]) * 0.1
        # LossRecord.append(loss.item())
        loss.backward()
        optimizer.step()
    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    return model
        



class first_1_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 7
        self.net = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode='reflect',
        )
    def forward(self,input):
        output1 = self.net(input)
        return output1
class first_2_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 21
        # self.net = nn.Conv1d(
        #     in_channels=1,
        #     out_channels=1,
        #     kernel_size=self.kernel_size,
        #     padding=self.kernel_size // 2,
        #     padding_mode='reflect',
        # )
        self.net = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                padding_mode='reflect',
            )
    def forward(self,input):
        output1 = self.net(input)
        return output1
class first_3_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 41
        self.net = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode='reflect',
        )
    def forward(self,input):
        output1 = self.net(input)
        return output1
class first_4_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.kernel_size = 61
        self.net = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode='reflect',
        )
    def forward(self,input):
        output1 = self.net(input)
        return output1
def trainfirstNet(*,model,data,lr=0.001,epoch=10):
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1.0)
    criterion = nn.MSELoss()
    LossRecord = []
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        output1 = model(data)
        loss = criterion(data, output1) * 1 \
               + torch.sum(torch.pow(model.net.weight, 2)) * 1 \
               + torch.sum(torch.abs(output1[:, :, 1:] - output1[:, :, :output1.shape[2] - 1])) * 1 \
               + torch.sum(output1[0][0] * output1[0][0]) * 0.01 \
               + torch.sum(output1[0][0]) * torch.sum(output1[0][0]) * 0.1
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    return model


def DifferenceL1(input):
    length = input.shape[-1]
    loss = 0
    for i in range(length - 1):
        loss = loss + torch.abs(input[0][0][i] - input[0][0][i + 1])
    return loss

# 二阶导数
def manifold(*,data,delay):
    length = data.shape[-1]
    data.view(1,length)
    a = data[0][:length-delay]
    b = data[0][delay:length]
    loss = 0
    for i in range(length-delay-3):
        loss = loss + pow(pow(a[i+2]+a[i]-2*a[i+1],2) + pow(b[i+2]+b[i]-2*b[i+1],2),0.5)
    return loss
# 信号能量
def energy(*,data,window_length=3):
    length = data.shape[-1]
    nums = data.shape[0]
    loss = 0
    for num in range(nums):
        for i in range(length-window_length-1):
            avg = torch.sum(data[num][i:i+window_length])/window_length
            for j in range(i,i+window_length):
                loss = loss + pow((data[num,j]-avg),2)
    return loss

