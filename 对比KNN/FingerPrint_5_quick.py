import torch

from Metric_learning_local import *
############################

torch.manual_seed(10)

class FingerPrint(nn.Module):
    def __init__(self, *, use_same_linear=False, input_data_dim, batches,
                 each_batch_dim):
        super(FingerPrint, self).__init__()

        # 必须保证头能被整除划分
        assert input_data_dim == batches * each_batch_dim

        self.use_same_linear = use_same_linear
        self.input_data_dim = input_data_dim

        self.batches = batches
        self.each_batch_dim = each_batch_dim

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
        self.classify = nn.Sequential(
            nn.Linear(self.input_data_dim, 1500, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(1500, self.each_batch_dim, bias=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, same_output):
        # 输入三条数据进行判别，每条数据的形状为 1 * persons
        # 最终输出结果为 each_batch_dim 维度
        ####
        same_output = same_output.unsqueeze(1).repeat(1, 1, 3)
        # output_data = torch.zeros((int(same_output.shape[0]), self.each_head_dim))  # .cuda()
        output_data = torch.zeros((int(same_output.shape[0]), 1, self.each_batch_dim)).cuda()
        qq = same_output[:, :, 0:self.input_data_dim]
        kk = same_output[:, :, self.input_data_dim:self.input_data_dim * 2]
        vv = same_output[:, :, 2 * self.input_data_dim:3 * self.input_data_dim]
        for i in range(self.batches):
            q = qq[:, :, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            k = kk[:, :, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            v = vv[:, :, i * self.each_batch_dim:(i + 1) * self.each_batch_dim]
            q = self.linear_transfer[3*i+0](q)
            k = self.linear_transfer[3*i+1](k)
            v = self.linear_transfer[3*i+2](v)
            # q = self.linear_transfer[0](q)
            # k = self.linear_transfer[1](k)
            # v = self.linear_transfer[2](v)
            output_data = torch.cat([output_data,
                                     torch.matmul(self.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.d_k),
                                                  v)], dim=-1)
        output_data = output_data[:, :, self.each_batch_dim:]
        output_data = self.combine_head_and_change_dim(output_data)

        # output_data = self.feed_forward(output_data)
        # output_data = self.softmax(output_data)
        output_data = self.classify(output_data)
        return output_data.squeeze(1)

def train_FingerPrint(*, model, data, label, target, lr=0.0001, epoch=2):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    # dataset = Data.TensorDataset(data, label, target)
    # loader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)
    criterion = nn.MSELoss()
    LossRecord = []
    length = data.shape[0]
    data = data.data
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        LossRecord.append(loss.item())
        optimizer.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    # plt.plot(LossRecord)
    # plt.show()
    # plt.savefig('D:/zqh/Image/FingerPrint_quick_loss')
    return model

def run_FingerPrint(epoch, Pathlist, batches):
    print('--------------指纹识别-------------------')
    oneperson_begin = 0
    oneperson_end = 20
    oneperson_nums = oneperson_end - oneperson_begin
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin, oneperson_end=oneperson_end)[:,:,:].cuda()
    # 区分设备
    # data = distinguish_device(data=data, device1=26, device2=16, onepersonnums=20)
    Unite_model = torch.load('/root/zqh/Save_Model/United_model_device.pth').cuda()
    feature1, ans, feature2 = Unite_model(data)
    features = feature2
    data = features
    persons = int(data.shape[0]/oneperson_nums)

    # 标签
    label = torch.zeros(oneperson_nums*persons)
    for i in range(persons):
        label[i*oneperson_nums:(i+1)*oneperson_nums] = i
    # 目标分类矩阵
    target = torch.zeros(oneperson_nums*persons, persons)
    for i in range(persons):
        target[i*oneperson_nums:(i+1)*oneperson_nums, i:(i+1)] = 2000

    data = data.cuda()

    # Metric_learning
    Metric_model = torch.load('/root/zqh/Save_Model/train_Metric_Model_local.pth').cuda()

    length = data.shape[0]
    ans = torch.zeros(length,length)

    output1 = Metric_model(data)  # 度量学习
    # output1 = data # 取消度量学习

    # 插入各类别权重
    weights = torch.zeros(data.shape[0], data.shape[-1]).cuda()
    weights[:, 0:data.shape[-1]] = 1
    # list = [3, 4, 5, 20, 24, 33]
    # list = [41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26]
    # list = [2, 4, 7, 10, 15, 16, 18, 29, 30, 31]
    # list = [5, 7, 8, 11, 12, 13, 14, 15, 19, 21, 23, 25, 29, 34, 37, 40,]  # KL散度
    # list = [11, 29, 34, 19, 40, 5, 23, 13, 7, 37, 14, 15, 25, 8, 33, 21]  # 最小的三个散度均值排序
    # list = [37, 32, 36, 34, 5, 16, 33, 24, 19, 29, 11, 2, 8, 40, 25, 20]  # 0703
    list = []
    # list = [37,32,36,34,5,16,33,24,19,29,11,2,20,10,40] # 0707
    for i in list:
        weights[i*int(data.shape[0]//label[-1]):(i+1)*int(data.shape[0]//label[-1]),150:200] = 1
    output1 = output1.squeeze(1)
    output1 = output1 * weights
    output1 = output1.cpu()

    muban = torch.zeros(persons,output1.shape[-1])
    # 通道一
    output = output1
    for i in range(persons):
        muban[i] = torch.mean(output[i*oneperson_nums:i*oneperson_nums+20, :], dim=0)
    ans = torch.zeros(data.shape[0],int(data.shape[0]//oneperson_nums))
    for i in range(data.shape[0]):
        sample = output[i].repeat(int(data.shape[0]//oneperson_nums), 1)
        save_dis = (sample - muban) * (sample - muban)
        save_dis = torch.sum(save_dis, dim=1)
        ans[i] = save_dis
    # 构建序列增强数据
    # batches = 4
    aans = torch.zeros(1,batches*ans.shape[-1]).cuda()
    from sklearn.model_selection import train_test_split
    for i in range(ans.shape[0]):
        # 随机生成batches-1条数据
        if batches==1:
            aans = torch.cat([aans.cuda(), ans.cuda()], dim=0)
            break
        else:
            databegin = i//oneperson_nums*oneperson_nums
            _, X_test, _, _ = train_test_split(ans[databegin:databegin+oneperson_nums,:],
                                               ans[databegin:databegin+oneperson_nums,:],
                                               test_size=(batches-1)/oneperson_nums,
                                               random_state=i)
            mid = torch.cat([ans[i:i+1,:].cuda(), X_test.cuda()],dim=0)
            mid = mid.view(1,batches*ans.shape[-1])
            aans = torch.cat([aans.cuda(), mid], dim=0)
    ans = aans[1:,:]
    ##################
    # ans 作为指纹输入
    ans = - ans.cuda()
    model = FingerPrint(input_data_dim=ans.shape[-1],batches=batches,each_batch_dim=int(ans.shape[-1]//batches)).cuda()
    model = train_FingerPrint(model=model, data=ans, target=target.cuda(),label=label.cuda(),lr=0.0001,epoch=epoch)
    torch.save(model, '/root/zqh/Save_Model/FingerPrint_quick_1.pth')
    print('模型保存成功！')

    # 查看结果
    right = 0
    record = torch.zeros(persons)
    output = model(ans)
    for i in range(ans.shape[0]):
        now_ans = torch.argmax(output[i])
        if now_ans == label[i]:
            record[now_ans] += 1
            right += 1
    print(right, '/', ans.shape[0])
    print(record)

# run_FingerPrint(2000)
