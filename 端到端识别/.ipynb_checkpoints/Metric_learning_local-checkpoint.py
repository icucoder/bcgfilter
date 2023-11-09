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
            nn.Linear(10000, 600, bias=False), # 原来50 0614修改
            # nn.LeakyReLU(inplace=True),
            # nn.Softmax(dim=2)
        )
        # 拉高度量学习维度，使得对比学习的输出结果可以将密度高的地方稀疏化（流形）

    def forward(self, output):
        output = self.linear(output)
        output = self.metric(output)
        return output