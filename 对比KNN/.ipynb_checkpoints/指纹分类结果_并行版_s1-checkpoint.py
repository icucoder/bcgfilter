from FingerPrint_5_quick import *
from United_model import *

def run_quick_test_ans(Pathlist, batches):
    print('测试集结果：')
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
    oneperson_end = 30
    focus_nums = 20 - oneperson_begin
    oneperson_nums = oneperson_end - oneperson_begin
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin, oneperson_end=oneperson_end)[:,:,:].cuda()
    # 区分设备
    # data = distinguish_device(data=data, device1=26, device2=16, onepersonnums=20)
    Unite_model = torch.load('/root/zqh/Save_Model/United_model_device.pth').cuda().eval()
    feature1, ans, feature2 = Unite_model(data)
    features = feature2
    data = features
    persons = int(data.shape[0]/oneperson_nums)

    # 标签
    label = torch.zeros(oneperson_nums*persons)
    for i in range(persons):
        label[i*oneperson_nums:(i+1)*oneperson_nums] = i
    # 目标分类矩阵
    target = torch.zeros(oneperson_nums*persons,persons)
    for i in range(persons):
        target[i*oneperson_nums:(i+1)*oneperson_nums, i:(i+1)] = 100

    data = data.cuda()

    # Metric_learning
    Metric_model = torch.load('/root/zqh/Save_Model/train_Metric_Model_local.pth').cuda().eval()

    length = data.shape[0]

    output1 = Metric_model(data)  # 度量学习
    # output1 = data # 取消度量学习

    # 插入各类别权重
    # weights = torch.zeros(data.shape[0], data.shape[-1]).cuda()
    # weights[:, 0:data.shape[-1]] = 1
    # list = [3, 4, 5, 20, 24, 33]
    # list = [41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26]
    # list = [2, 4, 7, 10, 15, 16, 18, 29, 30, 31]
    # list = [5, 7, 8, 11, 12, 13, 14, 15, 19, 21, 23, 25, 29, 34, 37, 40,]  # KL散度
    # list = [11, 29, 34, 19, 40, 5, 23, 13, 7, 37, 14, 15, 25, 8, 33, 21]  # 最小的三个散度均值排序
    # list = [37, 32, 36, 34, 5, 16, 33, 24, 19, 29, 11, 2, 8, 40, 25, 20]  # 0703
    # list = []
    # list = [37,32,36,34,5,16,33,24,19,29,11,2,20,10,40] # 0707
    # for i in list:
        # weights[i * int(label[-1]):(i + 1) * int(label[-1]), 200:256] = 1
        # weights[i * int(data.shape[0] // label[-1]):(i + 1) * int(data.shape[0] // label[-1]), 150:200] = 1
    output1 = output1.squeeze(1)
    # output1 = output1 * weights
    output1 = output1.cpu()

    muban = torch.zeros(persons, output1.shape[-1])
    # 通道一
    output = output1
    for i in range(persons):
        # muban[i] = torch.mean(output[i * oneperson_nums:i * oneperson_nums + 20, :], dim=0)
        muban[i] = torch.mean(output[i * oneperson_nums:i * oneperson_nums + focus_nums, :], dim=0)
    ans = torch.zeros(data.shape[0], int(data.shape[0] // oneperson_nums))
    for i in range(data.shape[0]):
        sample = output[i].repeat(int(data.shape[0] // oneperson_nums), 1)
        save_dis = (sample - muban) * (sample - muban) # L2范数
        # save_dis = torch.abs(sample - muban) # L1范数
        save_dis = torch.sum(save_dis, dim=1)
        ans[i] = save_dis

    print(ans.shape)
    print(label.shape)
    
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
            _, X_test, _, _ = train_test_split(ans[databegin+20:databegin+oneperson_nums,:],
                                               ans[databegin+20:databegin+oneperson_nums,:],
                                               test_size=(batches-1)/oneperson_nums,
                                               random_state=i)
            mid = torch.cat([ans[i:i+1,:].cuda(), X_test.cuda()],dim=0)
            mid = mid.view(1,batches*ans.shape[-1])
            aans = torch.cat([aans.cuda(), mid], dim=0)
    ans = aans[1:,:]
    ##################
    ans = -ans.cuda()
    model = torch.load('/root/zqh/Save_Model/FingerPrint_quick_1.pth').eval()
    print('模型读取成功！')
    # output = model(ans[1],ans[2],ans[3])
    # print(output)
    # print(target[1])

    # 五条连续序列测试结果
    right = 0
    record = torch.zeros(persons)
    output = model(ans)
    for i in range(ans.shape[0]):
        now_ans = torch.argmax(output[i])
        if now_ans == label[i]:
            record[now_ans] += 1
            right += 1

    for i in range(record.shape[0]):
        print(i,record[i])
    # print(record)
    print(right, '/', ans.shape[0], '=', right / ans.shape[0])
    print(right / ans.shape[0])

# run_quick_test_ans()