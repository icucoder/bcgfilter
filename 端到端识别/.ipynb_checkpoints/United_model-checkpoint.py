from TRM_Encoder_Decoder import *
from TRM_TRM_Unet import *
import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pylab as plt

torch.manual_seed(10)

class United_Model(nn.Module):
    def __init__(self,seq_length,elementlength): # 6 50
        super().__init__()
        self.elementlength = elementlength
        self.seq_length = seq_length
        self.data_length = seq_length * 100 # 需要与TRM输出特征维度一致
        self.model1 = Transformer(seq_length=seq_length, elementlength=elementlength).to(device)
        self.model2 = Transformer_Encoder(
            input_data_dim=self.data_length,
            batches=5,
            each_batch_dim=int(self.data_length/5),
            feed_forward_hidden_dim=20,
        ).to(device)


    def forward(self, input):
        input = input.view(input.shape[0], int(input.shape[2] / self.elementlength), self.elementlength).detach()
        features, ans = self.model1(input)
        features = features.view(features.shape[0], 1, features.shape[1] * features.shape[2])
        ans = ans.view(ans.shape[0], 1, ans.shape[1] * ans.shape[2])
        output = self.model2(features)
        return features, ans, output.squeeze(1)
