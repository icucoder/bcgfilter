# from ResUnet import *
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd
from Toolkit import *
import torch.utils.data as Data
import random
import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pylab as plt

torch.manual_seed(10)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # 分多头  要求能够整除
        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        # 分别对QKV做线性映射
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        # 对输出结果做线性映射
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    # 输入QKV以及Mask  得到编码结果
    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        # 把每个QKV中的单个分多头得到reshape后的QKV
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        # 对变形后的QKV做线性变换
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        # 计算得分
        attention = torch.softmax(energy / (self.embed_size ** (0.5)), dim=3).to(device)  # 240 5 10 10
        # 根据得分和value计算输出结果
        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)
        # 对输出结果做线性映射
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # 自注意力编码 输入为Q K V Mask
        self.attention = SelfAttention(embed_size, heads)
        # 归一化
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        # 前馈层
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    # 输入QKV Mask
    def forward(self, value, key, query):
        # 通过自注意力进行编码
        attention = self.attention(value, key, query)
        # 将  编码结果+Q   归一化
        x = self.dropout(self.norm1(attention + query))
        # 前馈层
        forward = self.feed_forward(x)
        # 再次进行归一化
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, embed_size, seq_length, elementlength=10):
        super(Encoder, self).__init__()
        self.seq_length = seq_length
        self.word_embedding = nn.Linear(elementlength, embed_size)
        self.position_embedding = nn.Embedding(seq_length, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size=embed_size, heads=8, dropout=0, forward_expansion=4, )
            for _ in range(6)]
        )
        # self.linear = nn.Linear(embed_size, 10)

    def forward(self, input):
        N = input.shape[0]
        seq_length = self.seq_length
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        p_embedding = self.position_embedding(positions)
        w_embedding = self.word_embedding(input)
        out = w_embedding + p_embedding
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key):
        attention = self.attention(x, x, x)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query)
        return out


class Decoder(nn.Module):
    def __init__(self, embed_size, seq_length=30, elementlength=10):
        super(Decoder, self).__init__()
        self.device = device
        self.seq_length = seq_length
        self.word_embedding = nn.Linear(elementlength, embed_size)
        self.position_embedding = nn.Embedding(seq_length, embed_size)  # 位置编码
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size=embed_size, heads=8, forward_expansion=4, dropout=0, device=device)
             for _ in range(6)]
        )
        self.fc_out = nn.Linear(embed_size, elementlength)
        self.dropout = nn.Dropout(0.1)

    def forward(self, enc_out):
        N = enc_out.shape[0]
        seq_length = self.seq_length
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(device)
        p_embedding = self.position_embedding(positions)
        w_embedding = enc_out
        x = (w_embedding + p_embedding)  # 向量 + 位置编码
        # DecoderBlock
        for layer in self.layers:
            x = layer(x, enc_out, enc_out)
        # 线性变换
        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            seq_length=30,
            elementlength=0,
            src_pad_idx=0,
            trg_pad_idx=0,
            num_layers=6,
            forward_expansion=4,
            heads=8,
            dropout=0,
            device="cuda",
            max_length=1000
    ):
        super(Transformer, self).__init__()
        self.embed_size = 256
        self.encoder = Encoder(embed_size=self.embed_size, seq_length=seq_length, elementlength=elementlength)
        self.decoder = Decoder(embed_size=self.embed_size, seq_length=seq_length, elementlength=elementlength)

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.linear = nn.Linear(self.embed_size, 100) # 需要与Unitedmodel的seqlength倍数一致
        # 降低对比学习输出特征维度
    def forward(self, src):  # shape  N 6 50
        # 编码
        enc_src = self.encoder(src)  # shape  N 6 256
        # 解码
        out = self.decoder(enc_src)  # shape  N 6 50
        features = self.linear(enc_src)
        return features, out

