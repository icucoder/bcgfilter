from ResUnet import *
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


def train_TRM_net(*, model, data, origin, target, elementlength, lr, epoch):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    LossRecord = []
    for k in tqdm(range(epoch)):
        optimizer.zero_grad()
        masked = data.clone()

        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2]-elementlength)
        masked[:, :, index:index + elementlength] = 0
        masked = masked.view(data.shape[0], int(data.shape[2]/elementlength), elementlength).detach()
        origin = origin.view(data.shape[0], int(data.shape[2]/elementlength), elementlength).detach()
        features, ans = model(masked)
        loss1 = criterion(ans, origin)

        def Simloss(ans, target):
            U = ans * target  # 行和为正样本
            V = ans  # 行和为所有样本
            FU = torch.exp(U)
            FV = torch.exp(V)
            Usum = torch.sum(FU, dim=1)
            Vsum = torch.sum(FV, dim=1)
            output = -torch.log(Usum / Vsum)
            return torch.sum(output, dim=0)

        features = features.view(data.shape[0], 1, features.shape[1] * features.shape[2])
        features = F.normalize(features).squeeze(1)
        loss2 = criterion(torch.mm(features, features.t()), target) * 10

        loss = loss1 + loss2
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    return model

class Classification(nn.Module):
    def __init__(self,input_data_dim,output_data_dim):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_data_dim,10000,bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10000,output_data_dim,bias=False),
            # nn.LeakyReLU(inplace=True),
            nn.Softmax(dim=2)
        )
    def forward(self,output):
        output = self.linear(output)
        return output

def run_classification(*,TRMmodel,classify_model,data,classify_label,lr,epoch):
    optimizer = optim.Adam(classify_model.parameters(), lr=lr, weight_decay=1.0)
    criterion = nn.MSELoss()
    LossRecord = []
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        output = TRMmodel(data)
        output = classify_model(output)
        loss = criterion(output,classify_label.unsqueeze(0))
        LossRecord.append(loss)
        loss.backward()
        optimizer.step()
    # LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    return classify_model

###############################################################
#

def run(epoch):
    Pathlist = [
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa1.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa5.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa6.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa7.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa8.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa9.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa10.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa12.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa13.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa14.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa16.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa18.pt',  # 新增数据
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa20.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa21.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa23.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa25.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa26.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa28.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa29.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa34.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa35.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa36.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa37.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa38.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa39.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_pa40.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_zqh1.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_zqh2.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_zzp612.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_tt612.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_whd612.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_qjf612.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_sjj612.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_zj612.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_dj613.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_dxt613.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_ltm613.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_rrx613.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_wg613.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_caoan615.pt',
        'D:/zqh/BCG125hz_Dataset/modify_extract_Single_resolution_sample1.pt',
    ]

    oneperson_begin = 0
    oneperson_end = 20
    oneperson_nums = oneperson_end - oneperson_begin
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin, oneperson_end=oneperson_end)
    data = data * 0.1
    persons = int(data.shape[0] / oneperson_nums)
    # 某条数据为某个人的标签
    label = torch.zeros(oneperson_nums * persons)
    for i in range(persons):
        label[i * oneperson_nums:(i + 1) * oneperson_nums] = i

    # 目标余弦相似度矩阵
    target = torch.zeros(oneperson_nums * persons, oneperson_nums * persons)
    for i in range(persons):
        target[i * oneperson_nums:(i + 1) * oneperson_nums, i * oneperson_nums:(i + 1) * oneperson_nums] = 1

    data = data[:,:,300:600]
    origin = data.clone()
    # index = random.randint(0, 75)
    # data[:,:,index:index+25] = 0
    # data = data.view(data.shape[0], 10, 10).cuda()

    elementlength = 50

    model = Transformer(seq_length=int(data.shape[2]/elementlength),elementlength=elementlength).to(device)

    # 随机掩码 掩码分成多个片段

    data = data.to(device)
    origin = origin.to(device)
    target = target.to(device)

    for i in range(1):
        model = train_TRM_net(model=model, data=data, origin=origin, target=target, elementlength=elementlength, lr=0.0005, epoch=epoch)

    masked = data.clone()
    index = random.randint(0, data.shape[2] - elementlength)
    masked[:, :, index:index + elementlength] = 0
    index = random.randint(0, data.shape[2] - elementlength)
    masked[:, :, index:index + elementlength] = 0
    index = random.randint(0, data.shape[2] - elementlength)
    masked[:, :, index:index + elementlength] = 0
    index = random.randint(0, data.shape[2] - elementlength)
    masked[:, :, index:index + elementlength] = 0
    index = random.randint(0, data.shape[2] - elementlength)
    masked[:, :, index:index + elementlength] = 0
    masked = masked.view(data.shape[0], int(data.shape[2] / elementlength), elementlength).detach()

    features, ans = model(masked.view(data.shape[0], int(data.shape[2]/elementlength), elementlength))

    ans = ans.view(data.shape[0], ans.shape[1]*ans.shape[2]).cpu()
    data = data.view(data.shape[0], data.shape[1]*data.shape[2]).cpu()
    masked = masked.view(masked.shape[0], masked.shape[1]*masked.shape[2]).cpu()
    torch.save(model, 'D:/zqh/BCG125hz_models/TRM_Unet_model_Singlefenbianlv_300.pth')
    print('模型保存成功！')

    for i in range(10):
        plt.subplot(10, 1, i + 1)
        if i == 0:
            plt.title('origin-blue masked-yellow ans-green')
        plt.plot(data[i].detach().numpy())
        plt.plot(masked[i].detach().numpy())
        plt.plot(ans[i].detach().numpy())
    plt.show()

    data = features.view(features.shape[0],features.shape[1]*features.shape[2])
    data = F.normalize(data)
    ans = torch.mm(data, data.t()).cpu()
    # ans = ans*ans
    map = plt.imshow(ans.detach().numpy(), interpolation='nearest', cmap=cm.Blues, aspect='auto', )
    plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
    plt.show()

# run(20000)

# 先尝试25 50 25   再尝试 300 片段

# model = Transformer().cuda()
#
# data = torch.rand(20,10,10).cuda()
#
# ans1, ans2 = model(data)
#
# print(ans1.shape)
# print(ans2.shape)