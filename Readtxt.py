import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('result_imputation.txt', sep='delimiter', header=None,engine='python') 

print(data.shape)
noise_list =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,5.0,10.0]
wjc ={}
wjc['mse'] = []
wjc['mae'] = []
wjc['name'] = 'Transformer'
Autoformer ={}
Autoformer['mse'] = []
Autoformer['mae'] = []
Autoformer['name'] = 'Autoformer'
TimesNet ={}
TimesNet['mse'] = []
TimesNet['mae'] = []
TimesNet['name'] = 'Timesnet'

for i in range(0,data.shape[0],2):
    string = data.at[i+1,0]
    split_string = string.split(", ")
    # 遍历分割后的字符串
    for item in split_string:
        # 判断字符串是否以指定的前缀开头
        if item.startswith("mse:"):
            mse = float(item.split(":")[1])
        elif item.startswith("mae:"):
            mae = float(item.split(":")[1])
    if 'wjc' in data.at[i,0]:
        name = wjc
    elif 'Autoformer' in data.at[i,0]:
        name = Autoformer
    else:
        name = TimesNet
    name['mse'].append(mse)
    name['mae'].append(mae)


for i in [wjc,Autoformer,TimesNet]:
    name = i['name']
    for j in range(1,5):
        mask_rate = str(0.125*j)
        path = 'pic_result/'
        plt.plot(noise_list[0:10],np.array(i['mse'])[j*13-13:j*13-3],label='mask_rate:'+mask_rate)
    # 添加图例
    plt.legend()
    plt.title(name+'_performance_with_SNR_under_mask_rate')
    plt.xlabel("SNR")
    plt.ylabel("mse")
    plt.savefig(path+name+'_'+"_mse_all.png")
    # 清空当前图形
    plt.clf()
    # 清空当前轴
    plt.cla()
    for j in range(1,5):
        mask_rate = str(0.125*j)
        plt.plot(noise_list[0:10],np.array(i['mae'])[j*13-13:j*13-3],label='mask_rate:'+mask_rate)
    # 添加图例
    plt.legend()
    plt.title(name+'_performance_with_SNR_under_mask_rate')
    plt.xlabel("SNR")
    plt.ylabel("mae")
    plt.savefig(path+name+'_'+"_mae_all.png")
    # 清空当前图形
    plt.clf()

    # 清空当前轴
    plt.cla()