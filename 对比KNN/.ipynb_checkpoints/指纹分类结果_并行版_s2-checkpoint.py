from FingerPrint_5_quick import *
from United_model import *

def splitDataSet(data, label, persons, oneperson_nums): # 要求输入 shape: N x
    X_train = data[0:20,:]
    X_test = data[20:30,:]
    y_train = label[0:20]
    y_test = label[20:30]
    for i in range(1,persons):
        X_train = torch.cat([X_train, data[oneperson_nums*i:oneperson_nums*i+20,:]],dim=0)
        X_test = torch.cat([X_test, data[oneperson_nums*i+20:oneperson_nums*i+30,:]],dim=0)
        y_train = torch.cat([y_train, label[oneperson_nums*i:oneperson_nums*i+20]],dim=0)
        y_test = torch.cat([y_test, label[oneperson_nums*i+20:oneperson_nums*i+30]],dim=0)
    return X_train, X_test, y_train, y_test

def run_quick_test_ans(Pathlist, batches):
    print('测试集结果：') 
    # 生成muban
    muban, label1 = create_Muban(Pathlist)
    
    oneperson_begin = 0
    oneperson_end = 30
    focus_nums = 20 - oneperson_begin
    oneperson_nums = oneperson_end - oneperson_begin
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin, oneperson_end=oneperson_end)[:,:,:].cuda()
    persons = int(data.shape[0]/oneperson_nums)
    # 标签
    label = torch.zeros(oneperson_nums*persons)
    for i in range(persons):
        label[i*oneperson_nums:(i+1)*oneperson_nums] = i
    
    X_train, X_test, y_train, y_test = splitDataSet(data, label, persons, oneperson_nums)
    # 对比学习
    Unite_model = torch.load('/root/zqh/Save_Model/United_model_device.pth').cuda().eval()
    feature1, ans, feature2 = Unite_model(X_test)
    features = feature2
    data = features
    data = data.cuda()
    # Metric_learning
    Metric_model = torch.load('/root/zqh/Save_Model/train_Metric_Model_local.pth').cuda().eval()
    output1 = Metric_model(data)  # 度量学习
    output1 = output1.squeeze(1)
    output1 = output1.cpu()
    # 通道一
    output = output1
    oneperson_nums = 10
    test_data = X_test
    ans = torch.zeros(test_data.shape[0], int(test_data.shape[0] // oneperson_nums))
    for i in range(test_data.shape[0]):
        sample = output[i].repeat(int(test_data.shape[0] // oneperson_nums), 1)
        save_dis = (sample - muban) * (sample - muban) # L2范数
        # save_dis = torch.abs(sample - muban) # L1范数
        save_dis = torch.sum(save_dis, dim=1)
        ans[i] = save_dis

    print('ans.shape=',ans.shape)
    
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
        if now_ans == y_test[i]:
            record[now_ans] += 1
            right += 1

    for i in range(record.shape[0]):
        print(i,record[i])
    # print(record)
    print(right, '/', ans.shape[0], '=', right / ans.shape[0])
    print(right / ans.shape[0])

# run_quick_test_ans()