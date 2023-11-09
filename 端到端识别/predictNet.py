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
        
        # self.resnet = nn.Linear(data_length, data_length)
        
        
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
        # 残差项 新增
        
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
        # 残差项 新增
        
        output1 = torch.nn.functional.normalize(output1, dim=1)
        output2 = torch.nn.functional.normalize(output2, dim=1)
        return output1.unsqueeze(1), output2.unsqueeze(1)
