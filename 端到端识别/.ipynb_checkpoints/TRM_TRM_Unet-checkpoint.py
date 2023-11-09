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


    def forward(self, same_output):

        same_output = same_output.repeat(1, 1, 3)
        # output_data = torch.zeros((int(same_output.shape[0]), self.each_head_dim))  # .cuda()
        output_data = torch.zeros((int(same_output.shape[0]), 1, self.each_batch_dim)).cuda()
        qq = same_output[:, :, 0:self.input_data_dim]
        kk = same_output[:, :, self.input_data_dim:self.input_data_dim * 2]
        vv = same_output[:, :, 2 * self.input_data_dim:3 * self.input_data_dim]
        for i in range(self.batches):
            q = qq[:,:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            k = kk[:,:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            v = vv[:,:, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            q = self.linear_transfer[3*i+0](q) # 240 1 320
            k = self.linear_transfer[3*i+1](k) 
            v = self.linear_transfer[3*i+2](v)
            # q = self.linear_transfer[0](q)
            # k = self.linear_transfer[1](k)
            # v = self.linear_transfer[2](v)
            att = torch.matmul(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.d_k), v)
            # print(output_data.shape)
            # print(att.shape)
            output_data = torch.cat([output_data,
                                     att],
                                    dim=-1)

        output_data = output_data[:,:, self.each_batch_dim:]
        output_data = self.combine_head_and_change_dim(output_data)

        # output_data = self.feed_forward(output_data)
        # output_data = self.softmax(output_data)
        return output_data

