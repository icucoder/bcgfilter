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
    def __init__(self, kernel_size1, kernel_size2, data_length):
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
        self.pred_length1 = kernel_size1//3
        self.pred_length2 = kernel_size2//3
        self.linear1 = nn.Linear(kernel_size1+1, self.pred_length1)
        self.linear2 = nn.Linear(kernel_size2+1, self.pred_length2)
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
        self.avglength = 2
        self.avgpool3 = nn.AvgPool1d(kernel_size=self.avglength, stride=self.avglength)
        self.avgpool4 = nn.AvgPool1d(kernel_size=self.avglength, stride=self.avglength)
        
        # self.resnet = nn.Linear(data_length, data_length)
        
        
    def forward(self, output):
        # input shape : 1 1 3210

        # 线性层        
        deckernel_size = 151
        avgpool1d = torch.nn.AvgPool1d(deckernel_size,stride=1,padding=deckernel_size//2)
        T = avgpool1d(output)
        output1 = output - T # 小核输入结果
        output1 = output1.squeeze(1) # 1 3210
        avg1 = self.avgpool1(output1) # 1 3210
        avg2 = self.avgpool2(output1)
        # 移位拼接
        output_mid1 = self.DIYroll(output1,self.kernel_size1) # k1  3210
        # 添加辅助趋势项
        output_mid1 = torch.cat([output_mid1, avg1],dim=0) # k1+1  3210
        # 线性变换 预测学习
        output11 = self.linear1(output_mid1.t()) # 3210 k1/3
        # 移位linear
        # output11 = output11.view(output.shape[-1]//(3*self.pred_length1), (3*self.pred_length1))
        
        output11 = output11[:,:self.avglength]
        # output11 = self.Avgpred(output11)
        output11 = torch.flatten(output11).unsqueeze(0)
        output11 = self.avgpool3(output11)
        # print(output11.shape) # 1 3210
        # print(output1.shape, output11.shape)
        output2 = output1 - output11 # 1 3210
        output_mid2 = self.DIYroll(output2,self.kernel_size2) # k2 3210
        output_mid2 = torch.cat([output_mid2, avg2],dim=0) # k2+1 3210
        output22 = self.linear2(output_mid2.t()) # 3210 k2/3
        # output22 = output22.view(output.shape[-1]//(3*self.pred_length2), (3*self.pred_length2))
        output22 = output22[:,:self.avglength]
        # output22 = self.Avgpred(output22)
        output22 = torch.flatten(output22).unsqueeze(0)
        output22 = self.avgpool3(output22)
        # print(output22.shape)
        output = output22+output11+T.squeeze(0)
        return output.unsqueeze(1)
    
    def Avgpred(input):
        m = input.shape[0]
        n = input.shape[1]
        output = torch.zeros(1, m)
        for i in range(0,m-n+1):
            output = torch.cat([output,
                               torch.cat([torch.zeros(1, i), input[i:i+1,:], torch.zeros(1, m-i)],dim=1)
                               ],dim=0)
        output = output[1:,:]
        output = torch.mean(output, dim=0)
        return output
        
    def DIYroll(self, input, shift):
        output = torch.roll(input,shifts=-1,dims=1)
        for i in range(2,shift+1):
            output = torch.cat([output, 
                                torch.roll(input,shifts=i,dims=1),
                                ],dim=0)
        return output
    
    def returnAns(self, output):
        # input shape : N 1 10000
        # 线性层     
        deckernel_size = 151
        avgpool1d = torch.nn.AvgPool1d(deckernel_size,stride=1,padding=deckernel_size//2)
        T = avgpool1d(output)
        output1 = output - T # 小核输入结果
        output1 = output1.squeeze(1) # 1 3210
        avg1 = self.avgpool1(output1) # 1 3210
        avg2 = self.avgpool2(output1)
        # 移位拼接
        output_mid1 = self.DIYroll(output1,self.kernel_size1) # k1  3210
        # 添加辅助趋势项
        output_mid1 = torch.cat([output_mid1, avg1],dim=0) # k1+1  3210
        # 线性变换 预测学习
        output11 = self.linear1(output_mid1.t()) # 3210 k1/3
        # 移位linear
        # output11 = output11.view(output.shape[-1]//(3*self.pred_length1), (3*self.pred_length1))
        output11 = output11[:,:1]
        output11 = torch.flatten(output11.t()).unsqueeze(0)
        # print(output11.shape) # 1 3210
        # print(output1.shape, output11.shape)
        # output2 = output1 # - output11 # 1 3210 # 串行
        output2 = output1 # 并行
        output_mid2 = self.DIYroll(output2,self.kernel_size2) # k2 3210
        output_mid2 = torch.cat([output_mid2, avg2],dim=0) # k2+1 3210
        output22 = self.linear2(output_mid2.t()) # 3210 k2/3
        # output22 = output22.view(output.shape[-1]//(3*self.pred_length2), (3*self.pred_length2))
        output22 = output22[:,:1]
        output22 = torch.flatten(output22.t()).unsqueeze(0)
        # print(output22.shape)
        output = output22+output11+T.squeeze(0)
        return output11.unsqueeze(1), output22.unsqueeze(1)
    
    

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

