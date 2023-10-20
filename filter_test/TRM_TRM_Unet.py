import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd
from Toolkit import *
import torch.utils.data as Data

class Transformer_Encoder(nn.Module):
    def __init__(self, *, use_same_linear=False, input_data_dim, batches,
                 each_batch_dim, feed_forward_hidden_dim):
        super(Transformer_Encoder, self).__init__()

        # 必须保证头能被整除划分
        assert input_data_dim == batches * each_batch_dim

        self.use_same_linear = use_same_linear
        self.input_data_dim = input_data_dim

        self.batches = batches
        self.each_batch_dim = each_batch_dim
        self.feed_forward_hidden_dim = feed_forward_hidden_dim

        self.d_k = each_batch_dim ** -0.5

        self.softmax = nn.Softmax(dim=-1)
        # 是否采用同一个线性映射得到q、k、v
        self.linear_transfer = nn.Linear(self.input_data_dim,
                                         self.mid_data_dim) \
            if self.use_same_linear else nn.ModuleList([nn.Linear(self.each_batch_dim,
                                                                  self.each_batch_dim) for _ in range(99)])
        # 如果使用多头注意力将其降低为原来的通道
        self.combine_head_and_change_dim = nn.Linear(self.batches * self.each_batch_dim,
                                                     self.input_data_dim)

        # 向前连接层  无效
        # self.feed_forward = nn.Sequential(
        #     nn.Linear(self.input_data_dim, self.feed_forward_hidden_dim),
        #     # nn.GELU(),
        #     # nn.Dropout(0.1),
        #     # nn.Linear(self.feed_forward_hidden_dim, input_data_dim),
        #     nn.Linear(self.feed_forward_hidden_dim, input_data_dim),
        #     # nn.Dropout(0.1)
        # )

    def forward(self, same_output):

        same_output = same_output.repeat(1, 1, 3)
        # output_data = torch.zeros((int(same_output.shape[0]), self.each_head_dim))  # .cuda()
        output_data = torch.zeros((int(same_output.shape[0]), self.each_batch_dim)).cuda()
        qq = same_output[:, :, 0:self.input_data_dim].squeeze(1)
        kk = same_output[:, :, self.input_data_dim:self.input_data_dim * 2].squeeze(1)
        vv = same_output[:, :, 2 * self.input_data_dim:3 * self.input_data_dim].squeeze(1)
        for i in range(self.batches):
            q = qq[:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            k = kk[:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            v = vv[:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            q = self.linear_transfer[3*i+0](q) # 240 1 320
            k = self.linear_transfer[3*i+1](k) 
            v = self.linear_transfer[3*i+2](v)
            # q = self.linear_transfer[0](q)
            # k = self.linear_transfer[1](k)
            # v = self.linear_transfer[2](v)
            output_data = torch.cat([output_data,
                                     torch.matmul(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.d_k),
                                                  v)], dim=-1)

        output_data = output_data[:, self.each_batch_dim:]
        output_data = self.combine_head_and_change_dim(output_data)

        # output_data = self.feed_forward(output_data)
        # output_data = self.softmax(output_data)
        return output_data.unsqueeze(1)


# 无batch_size  target表示最终的目标余弦相似度矩阵  label表示某一个人的数据
def train_TRM_Model(*,model,data,label,target,lr=0.0001,epoch=2):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    criterion = nn.MSELoss()
    LossRecord = []
    length = data.shape[0]

    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        zero = torch.tensor(0.0).cuda()
        loss = 0
        # ans = torch.zeros(length,length).cuda()
        # 20230506修改
        output = model(data).squeeze(1)
        output = F.normalize(output)
        sim_ans = torch.mm(output,output.t())
        # loss = criterion(sim_ans,target)
        def Simloss(ans,target):
            U = ans * target # 行和为正样本
            V = ans # 行和为所有样本
            FU = torch.exp(U)
            FV = torch.exp(V)
            Usum = torch.sum(FU,dim=1)
            Vsum = torch.sum(FV,dim=1)
            output = -torch.log(Usum/Vsum)
            return torch.sum(output,dim=0)
        loss = Simloss(sim_ans,target)
        # 20230506之前  5行
        # for i in range(length):
        #     for j in range(length):
        #         if label[i] == label[j]:
        #             loss = loss + 1 - C_similarity(model(data[i])[0][0],model(data[j])[0][0])
        #         else:
        #             loss = loss + criterion(C_similarity(model(data[i])[0][0], model(data[j])[0][0]),zero)
                
                # ans[i][j] = torch.cosine_similarity(model(data[i]),model(data[j]),dim=2)
                # loss = loss + criterion(C_similarity(model(data[i])[0][0],model(data[j])[0][0]),label[i][j])
                # ans[i][j] = C_similarity(model(data[i])[0][0],model(data[j])[0][0])
        # output = model(data)
        # for i in range(length):
        #     for j in range(length):
        #         if label[i] == label[j]:
        #             loss = loss + 1 - C_similarity(output[i][0],output[j][0])
        #         else:
        #             loss = loss + criterion(C_similarity(output[i][0],output[j][0]),zero)

        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    return model

# 更改label  有batch_size
# def train_TRM_Model(*,model,data,label,lr=0.0001,epoch=2):
#     dataset = Data.TensorDataset(data,label)
#     loader = Data.DataLoader(dataset=dataset,batch_size=10,shuffle=True)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
#     criterion = nn.MSELoss()
#     LossRecord = []
#     length = data.shape[0]
#
#     for _ in tqdm(range(epoch)):
#         optimizer.zero_grad()
#         # ans = torch.zeros
#         loss = 0
#         # ans = torch.zeros(length,length).cuda()
#         zero = torch.tensor(0.0).cuda()
#         for step,(dataa,labell) in enumerate(loader):
#             for i in range(dataa.shape[0]):
#                 for j in range(dataa.shape[0]):
#                     if labell[i] == labell[j]:
#                         loss = loss + 1 - C_similarity(model(dataa[i])[0][0],model(dataa[j])[0][0])
#                     else:
#                         loss = loss + criterion(C_similarity(model(dataa[i])[0][0], model(dataa[j])[0][0]),zero)
#                 # ans[i][j] = C_similarity(model(data[i])[0][0],model(data[j])[0][0])
#
#         LossRecord.append(loss)
#         loss.backward()
#         optimizer.step()
#     LossRecord = torch.tensor(LossRecord, device="cpu")
#     plt.plot(LossRecord)
#     plt.show()
#     return model


class Classification(nn.Module):
    def __init__(self,input_data_dim,output_data_dim):
        super().__init__()
        # self.conv = nn.Conv1d(1,1,kernel_size=21,padding=10)
        # self.linear = nn.Linear(input_data_dim,output_data_dim)
        self.linear = nn.Sequential(
            nn.Linear(input_data_dim,200,bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200,output_data_dim,bias=False),
            nn.LeakyReLU(inplace=True),
            # nn.Softmax(dim=2)
        )
    def forward(self,output):
        # output = self.conv(output)
        output = self.linear(output)
        return output

def run_classification(*,TRMmodel,classify_model,data,classify_label,lr,epoch):
    optimizer = optim.Adam(classify_model.parameters(), lr=lr, weight_decay=1.0)
    criterion = nn.MSELoss()
    LossRecord = []
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        loss = 0
        # for i in range(data.shape[0]):
        #     output = classify_model(TRMmodel(data[i]))
        #     loss = loss + criterion(output,classify_label[i].unsqueeze(0))
        output = TRMmodel(data)
        output = classify_model(output)
        loss = criterion(output,classify_label.unsqueeze(0))
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return classify_model