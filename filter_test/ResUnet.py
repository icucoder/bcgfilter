import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd
from Toolkit import *
import torch.utils.data as Data
import random

torch.manual_seed(1.0)
torch.cuda.manual_seed_all(1.0)

class Conv(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3):
        super().__init__()
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                # padding_mode='reflect',
            ),
            # nn.BatchNorm1d(self.out_channels),
            nn.Conv1d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                # padding_mode='reflect',
            ),
        )
    def forward(self,x):
        return self.layer(x)

class ResNetDown(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=21):
        super().__init__()
        self.stride = 2
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            Conv(in_channels, out_channels),
            Conv(out_channels, out_channels),
        )
        self.layer = nn.Sequential(
            nn.Conv1d(out_channels,out_channels,2,self.stride),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, self.kernel_size, padding=self.kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
    def forward(self,output):
        same_output = self.conv(output)
        output = self.layer(same_output)
        return output

class Up(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = 30
        self.up = nn.ConvTranspose1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=2,
                padding=(self.kernel_size-1) // 2,
            )
        self.conv = Conv(in_channels,out_channels)
    def forward(self,x1,x2):
        x1 = self.up(x1)
        x = torch.cat([x1,x2],dim=1)
        x = self.conv(x)
        return x
# 把卷积更换成MAE
class ResUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = Conv(1,64)
        self.down1 = ResNetDown(64, 128)
        self.down2 = ResNetDown(128, 256)
        self.up1 = Up(256,128)
        self.up2 = Up(128, 64)
        self.outc = Conv(64, 1)

        self.combine1 = nn.Sequential(
            Conv(128, 1),
            nn.BatchNorm1d(1),
        )
        self.combine2 = nn.Sequential(
            Conv(256, 1),
            nn.BatchNorm1d(1),
        )
        self.linear = nn.Linear(75,75)

    def forward(self,output):
        # MAE
        def MAE(data):
            length = data.shape[2]
            mask_len = 25
            zeros = torch.zeros(data.shape[0],data.shape[1],mask_len)
            index = random.randint(1,length - mask_len)
            ans = torch.cat([data[:,:,:index],zeros],dim=2)
            ans = torch.cat([ans,data[:,:,index+mask_len:]],dim=2)
            return ans
        output = MAE(output).detach()
        # 下采样
        output = self.inc(output)   # 1 1 100 -> 1 64 100
        output1 = self.down1(output)   # 1 64 100 -> 1 128 50          下采样特征
        output2 = self.down2(output1)   # 1 128 50 -> 1 256 25         下采样特征
        # 添加噪声
        noise = 0.001*torch.randn(output2.shape)
        output2 = output2 + noise
        # 上采样
        output3 = self.up1(output2,output1)   # 1 256 25 -> 1 128 50
        output4 = self.up2(output3,output)   # 1 128 50 -> 1 64 100
        output = self.outc(output4)   # 1 64 100 -> 1 1 100
        # cat features
        feature1 = self.combine1(output1)
        feature2 = self.combine2(output2)
        features = torch.cat([feature1,feature2],dim=2)
        # linear
        features = self.linear(features)
        del feature1,feature2,output1,output2,noise,output3,output4
        return features, output


def train_ResUnet(*,model,data,label,lr,epoch):
    # dataset = Data.TensorDataset(data, label)
    # loader = Data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    LossRecord = []
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        output1, output2 = model(data)
        loss1 = criterion(output2, data)
        loss2 = torch.tensor(0.0)
        loss3 = torch.tensor(0.0)
        for i in range(output1.shape[0]):
            for j in range(output1.shape[0]):
                if label[i] == label[j]:
                    loss2 = loss2 + criterion(output1[i],output1[j])
                    loss3 = loss3 + criterion(output2[i],output2[j])

        loss = loss1*10 + loss2*10 + loss3*0
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    return model

# 将不同分辨率下的数据投影到同一度量空间
def train_ResUnet2(*,model,data,label,target,lr,epoch):
    dataset = Data.TensorDataset(data, label)
    batch_size = 10
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    LossRecord = []
    for k in tqdm(range(epoch)):
        optimizer.zero_grad()

        # 有batch size
        # output11, output12 = model(data[:,:,0:100])
        # output21, output22 = model(data[:, :, 100:200])
        # output31, output32 = model(data[:, :, 200:300])
        # output1 = torch.cat([output11,output21,output31],dim=2)
        # output2 = torch.cat([output12,output22,output32],dim=2)
        # loss1 = criterion(output2, data)
        # del output1,output2,output11,output12,output21,output22,output31,output32
        # loss3 = torch.tensor(0.0)
        # for step,(databatch,labelbatch) in enumerate(loader):
        #     output11, output12 = model(databatch[:, :, 0:100])
        #     output21, output22 = model(databatch[:, :, 100:200])
        #     output31, output32 = model(databatch[:, :, 200:300])
        #     for i in range(batch_size):
        #         for j in range(batch_size):
        #             if label[i] == label[j]:
        #                 loss3 = loss3 + criterion(output11[i],output21[j]) + criterion(output11[i],output31[j]) + criterion(output21[i],output31[j])
        #             if label[i] != label[j]:
        #                 loss3 = loss3 + abs(C_similarity(output11[i],output11[j])) + abs(C_similarity(output21[i],output21[j])) + abs(C_similarity(output31[i], output31[j]))

        # 无batch size
        output11, output12 = model(data[:, :, 0:100])
        output21, output22 = model(data[:, :, 100:200])
        output31, output32 = model(data[:, :, 200:300])
        output1 = torch.cat([output11,output21,output31],dim=2)
        output2 = torch.cat([output12,output22,output32],dim=2)
        del output12,output22,output32
        loss1 = criterion(output2, data)
        del output2
        loss3 = torch.tensor(0.0)
        for i in range(output1.shape[0]):
            for j in range(output1.shape[0]):
                if label[i] == label[j]:
                    # 正样本拉近
                    loss31 = criterion(output11[i], output21[j])
                    loss32 = criterion(output11[i], output31[j]) 
                    loss33 = criterion(output21[i], output31[j])
                    loss3 = loss3 + loss31 + loss32 +loss33
                    del loss31,loss32,loss33
                if label[i] != label[j]:
                    # 负样本拉远
                    loss3 = loss3 + (abs(C_similarity(output11[i], output11[j])) 
                                     + abs(C_similarity(output21[i], output21[j])) 
                                     + abs(C_similarity(output31[i], output31[j])))
        del output11,output21,output31
        loss = loss1 + loss3
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del loss,loss1,loss3
        # torch.cuda.empty_cache()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    print(LossRecord[LossRecord.shape[-1]-1])
    # plt.plot(LossRecord)
    # plt.show()
    return model



def train_ResUnet3(*,model,data,label,target,lr,epoch):
    dataset = Data.TensorDataset(data, label)
    batch_size = 10
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    LossRecord = []
    for k in tqdm(range(epoch)):
        optimizer.zero_grad()
        # 掩码
        # def MAE(data):
        #     length = data.shape[2]
        #     mask_len = 25
        #     index = random.randint(1,length - mask_len)
        #     data[:,:,index:index+mask_len] = 0
        #     return data
        # data = MAE(data).detach()
        # 无batch size
        output11, output12 = model(data[:, :, 0:100])
        output21, output22 = model(data[:, :, 100:200])
        output31, output32 = model(data[:, :, 200:300])
        output1 = torch.cat([output11,output21,output31],dim=2)
        output2 = torch.cat([output12,output22,output32],dim=2)
        del output12,output22,output32
        loss1 = criterion(output2, data)
        del output2
        loss3 = torch.tensor(0.0)
        output11 = F.normalize(output11).squeeze(1)
        output21 = F.normalize(output21).squeeze(1)
        output31 = F.normalize(output31).squeeze(1)
        # ans1 = criterion(torch.mm(output11,output11.t()),target)
        # ans2 = criterion(torch.mm(output11,output21.t()),target)
        # ans3 = criterion(torch.mm(output11,output31.t()),target)
        # ans4 = criterion(torch.mm(output21,output21.t()),target)
        # ans5 = criterion(torch.mm(output21,output31.t()),target)
        # ans6 = criterion(torch.mm(output31,output31.t()),target)
        ans1 = torch.mm(output11,output11.t())
        # ans2 = torch.mm(output11,output21.t())
        # ans3 = torch.mm(output11,output31.t())
        ans4 = torch.mm(output21,output21.t())
        # ans5 = torch.mm(output21,output31.t())
        ans6 = torch.mm(output31,output31.t())
        def Simloss(ans,target):
            U = ans * target # 行和为正样本
            V = ans # 行和为所有样本
            FU = torch.exp(U)
            FV = torch.exp(V)
            Usum = torch.sum(FU,dim=1)
            Vsum = torch.sum(FV,dim=1)
            output = -torch.log(Usum/Vsum)
            return torch.sum(output,dim=0)
        # loss3 = Simloss(ans4,target)+Simloss(ans1,target)+Simloss(ans2,target)+Simloss(ans3,target)+Simloss(ans5,target)+Simloss(ans6,target)
        # loss3 = criterion(ans4*ans4,target)
        # ans = torch.abs(ans1)+torch.abs(ans2)+torch.abs(ans3)+torch.abs(ans4)+torch.abs(ans5)+torch.abs(ans6)
        # ans = ans1*ans1 + ans2*ans2 + ans3*ans3 + ans4*ans4 + ans5*ans5 + ans6*ans6 
        # ans = ans1*ans1 + ans4*ans4 + ans6*ans6
        # loss3 = criterion(ans,target*3)
        del output11,output21,output31
        loss1 = loss1 * 0.1
        loss3 = 0
        # print(loss1,loss3)
        loss = loss3 + loss1
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        del loss,loss1,loss3
        # torch.cuda.empty_cache()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model


####################################################################
# Transformer 编码解码结构

class TRM_Encoder(nn.Module):
    def __init__(self, *, use_same_linear=False, input_data_dim, batches,
                 each_batch_dim, feed_forward_hidden_dim):
        super(TRM_Encoder, self).__init__()

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
            q = self.linear_transfer[3*i+0](q)
            k = self.linear_transfer[3*i+1](k)
            v = self.linear_transfer[3*i+2](v)
            output_data = torch.cat([output_data,
                                     torch.matmul(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.d_k),
                                                  v)], dim=-1)

        output_data = output_data[:, self.each_batch_dim:]
        output_data = self.combine_head_and_change_dim(output_data)

        # output_data = self.feed_forward(output_data)
        # output_data = self.softmax(output_data)
        return output_data.unsqueeze(1)

# https://blog.csdn.net/qq_38406029/article/details/122050257  带有掩码的Tranformer
