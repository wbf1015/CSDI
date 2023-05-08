import numpy as np
import pandas as pd
import csv
import random
import sys
import torch

missing_data_path = '/home/rst05/codes/clustering/dataset/hcv/miss_hcv_for_csdi.csv'
trans_data_path = '/home/rst05/codes/csdi/data/plrx/plrxA/'

def fill_missing_values(X):
    """
    X: numpy.array类型，包含缺失值
    """
    # 找到所有的缺失值
    missing = X == -200
    # 将缺失值替换成0，以便在计算列均值时不影响结果
    X[missing] = 0
    # 计算每一列的平均值
    column_means = np.nanmean(X, axis=0)
    # 将缺失值用每列的平均值进行替换
    X[missing] = np.take(column_means, np.where(missing)[1])
    return X

def transGTdata():
    missing_data = pd.read_csv(missing_data_path, delimiter=",", index_col=None, header=None).to_numpy()
    header = missing_data[0,:]
    missing_data = missing_data[1:,:]
    missing_data = missing_data.astype('float64')
    missing_data = fill_missing_values(missing_data)
    
    random.seed()
    Time = '00:00'
    new_header = ['Time','Parameter','Value']
    r = random.randint(100000,200000)
    
    for data in missing_data:
        with open(trans_data_path + str(r) + '.txt', 'w', newline='') as file_obj:
            writer = csv.writer(file_obj)
            writer.writerow(new_header)
            count = 0
            for attr in data:
                l = []
                l.append(Time)
                l.append(header[count])
                l.append(attr)
                writer.writerow(l)
                count += 1
        r += 1
    
def getGTmask(origin_gtmask):
    # print(origin_gtmask)
    # print(origin_gtmask.shape)
    tuple_num = origin_gtmask.shape[0]
    attr_num = origin_gtmask.shape[2]
    
    new_gtmasks = []
    missing_data = pd.read_csv(missing_data_path).to_numpy()
    for data in missing_data:
        new_gtmask = []
        gt_mask = [1] * attr_num
        for i in range(len(data)):
            if data[i] == -200:
                gt_mask[i] = 0
        new_gtmask.append(gt_mask)
        new_gtmasks.append(new_gtmask)
    
    new_gtmasks = np.array(new_gtmasks)
    # print(new_gtmasks)
    # print(new_gtmasks.shape)
        
    return new_gtmasks
    
def getLoaderIndex():
    missing_data = pd.read_csv(missing_data_path).to_numpy()
    test_index = []
    temp_train_index = []
    for i in range(len(missing_data)):
        is_lost = False
        for j in range(len(missing_data[0])):
            if missing_data[i][j] == -200:
                is_lost = True
        if is_lost:
            test_index.append(i)
        else:
            temp_train_index.append(i)   
    
    np.random.shuffle(temp_train_index)
    num_train = (int)(len(missing_data) * 0.7)
    train_index = temp_train_index[:num_train]
    valid_index = temp_train_index[num_train:] 
    return train_index,valid_index,test_index

def inverse_trans(c_target2,samples2):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    origin_path = missing_data_path
    org_data = pd.read_csv(origin_path).to_numpy()
    org_data = fill_missing_values(org_data)
    stand_scaler = StandardScaler()
    std_data = stand_scaler.fit_transform(org_data)
    c_target = c_target2.cpu().numpy()
    c_target = stand_scaler.inverse_transform(c_target)
    c_target = torch.from_numpy(c_target)
    c_target = c_target.to('cuda:0')
    
    samples = samples2.cpu().numpy()
    samples = stand_scaler.inverse_transform(samples)
    samples = torch.from_numpy(samples)
    samples = samples.to('cuda:0')
    return c_target ,samples

def getimputed_data(imputed_data,ground_truth_data,location):
    imputed_data = imputed_data.cpu().numpy()
    ground_truth_data = ground_truth_data.cpu().numpy()
    location = location.cpu().numpy()
    print(imputed_data.shape)
    print(ground_truth_data.shape)
    print(location.shape)
    store_data = []
    for i in range(location.shape[0]):
        for j in range(location.shape[1]):
            data = ground_truth_data[i][j]
            has_nan = False
            for k in range(location.shape[2]):
                if location[i][j][k].all() == 1:
                    data[k] = imputed_data[i][j][k].all()
                    has_nan = True
            if has_nan:
                store_data.append(data)
    
    for data in store_data:
        data = data.tolist()
        print(data)
    
    with open('hcvresult', 'a', newline='') as file_obj:
        writer = csv.writer(file_obj)
        for data in store_data:
            writer.writerow(data)
    
    

if __name__ == "__main__":
    transGTdata()
    
    
    
    
    
    