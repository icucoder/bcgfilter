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
from United_model import *
from Toolkit import *

# 50维 只取20维做度量学习
# 将整个维度分为两份 对分别两份进行度量学习

torch.manual_seed(10)

class Metric_Model(nn.Module):
    def __init__(self, input_data_dim):
        super(Metric_Model, self).__init__()
        self.input_data_dim = input_data_dim
        self.linear = nn.Linear(input_data_dim,input_data_dim)
        self.metric = nn.Sequential(
            nn.Linear(self.input_data_dim, 10000, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10000, 55, bias=False), # 原来50 0614修改
            nn.LeakyReLU(inplace=True),
            nn.Softmax(dim=1)
        )
        # 拉高度量学习维度，使得对比学习的输出结果可以将密度高的地方稀疏化（流形）

    def forward(self, output):
        output = self.linear(output)
        output = self.metric(output)
        return output

# 引入流形

def train_Metric_Model(*, model, data, label, target, lr=0.0001, epoch=2):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    dataset = Data.TensorDataset(data, label)
    loader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)
    criterion = nn.MSELoss()
    LossRecord = []
    length = data.shape[0]
    data = data.data
    # 插入各类别权重
    # list = [3,4,5,20,24,33]
    # list = [41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26]
    # list = [2, 4, 7, 10, 15, 16, 18, 29, 30, 31]
    # list = [5, 7, 8, 11, 12, 13, 14, 15, 19, 21, 23, 25, 29, 34, 37, 40,] # KL散度
    # list = [11, 29, 34, 19, 40, 5, 23, 13, 7, 37, 14, 15, 25, 8, 33, 21]  # 最小的三个散度均值排序
    # list = [37,32,36,34,5,16,33,24,19,29,11,2,8,40,25,20] # 0703
    # list = []
    # list = [37,32,36,34,5,16,33,24,19,29,11,2,20,10,40] # 0707
    # for i in list:
    #     weights[i*int(data.shape[0]//label[-1]):(i+1)*int(data.shape[0]//label[-1]),150:200] = 1
    
    newlabel = label.cuda()
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        zero = torch.tensor(0.0).cuda()
        output = model(data)
        output = output.squeeze(1)
        loss = criterion(output.cuda(), label.cuda())
        loss.backward()
        optimizer.step()
        LossRecord.append(loss.item())
    plt.show()
    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    # plt.savefig('D:/zqh/Image/Metric_learning_loss')
    return model



def Simloss(ans, target):
    U = ans * target  # 行和为正样本
    V = ans  # 行和为所有样本
    FU = torch.exp(U)
    FV = torch.exp(V)
    Usum = torch.sum(FU, dim=1)
    Vsum = torch.sum(FV, dim=1)
    output = -torch.log(Usum / Vsum)
    return torch.sum(output, dim=0)



############################


def run_Metric_Model(epoch, Pathlist):
    # 放入全部人员
    # Pathlist = [
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa1.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa5.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa6.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa7.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa8.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa9.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa10.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa11.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa12.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa13.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa14.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa15.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa16.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa17.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa18.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa19.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa20.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa21.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa22.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa23.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa24.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa25.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa26.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa27.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa28.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa29.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa30.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa31.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa32.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa33.pt',  # 新增数据
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa34.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa35.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa36.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa37.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa38.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa39.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_pa40.pt',  # 26人
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_zqh1.pt',
    #     # 'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_zqh2.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_zzp612.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_tt612.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_whd612.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_qjf612.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_sjj612.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_zj612.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_dj613.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_dxt613.pt',
    #     # 'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_ltm613.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_rrx613.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_wg613.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_wxy.pt',
    #     # 'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_caoan615.pt',
    #     '/root/zqh/BCGDataSet/modify_extract_Single_resolution_sample1.pt',  # 16人
    # ]
    # 新增数据
    # Pathlist = [
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa15.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa17.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa18.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa19.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa22.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa27.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa30.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa31.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa32.pt',  # 新增数据
    #     'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa33.pt',  # 新增数据
    # ]
    oneperson_begin = 0
    oneperson_end = 20
    oneperson_nums = oneperson_end - oneperson_begin
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin, oneperson_end=oneperson_end)[:,:,:].cuda()

    # 区分设备
    # data = distinguish_device(data=data, device1=26, device2=16, onepersonnums=20)

    Unite_model = torch.load('/root/zqh/Save_Model/United_model_device.pth').cuda().eval()
    feature1, ans, feature2 = Unite_model(data)
    features = feature2 # 对比学习
    # features = ans # 对比学习取消
    # features = feature1 # 对比学习取消
    data = features
    persons = int(data.shape[0]/oneperson_nums)

    # 标签
    label = torch.zeros(oneperson_nums*persons)
    for i in range(persons):
        label[i*oneperson_nums:(i+1)*oneperson_nums] = i
    # 目标余弦相似度矩阵
    target = torch.zeros(oneperson_nums*persons,oneperson_nums*persons)
    for i in range(persons):
        target[i*oneperson_nums:(i+1)*oneperson_nums, i*oneperson_nums:(i+1)*oneperson_nums] = 1
    # 新标签
    newlabel = torch.zeros(oneperson_nums*persons, persons)
    for i in range(persons):
        newlabel[i*oneperson_nums:(i+1)*oneperson_nums, i:(i+1)] = 1

    data = data.cuda()
    # label = label.cuda()
    target = target.cuda()

    # Metric_learning
    print('--------------度量学习-------------------')
    print('features shape :',features.shape)
    model = Metric_Model(input_data_dim=features.shape[-1]).cuda()
    model = train_Metric_Model(model=model,data=features,label=newlabel,target=target,lr=0.0001,epoch=epoch)
    torch.save(model,'/root/zqh/Save_Model/train_Metric_Model_local.pth')
    print('模型保存成功！')

    length = data.shape[0]
    ans = torch.zeros(length,length)

    output1 = model(data)
    print(output1)
    # plt.show()
    # plt.savefig('D:/zqh/Image/Metric_learning_ans')

# run_Metric_Model(1000)