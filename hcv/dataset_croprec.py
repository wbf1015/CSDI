import pickle
import sys
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from csdi_imputation import getGTmask,getLoaderIndex

# 35 attributes which contains enough non-values
# 感觉像是phsio里面的属性，但是不全
attributes = ['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','A11','A12']

# 返回小时数
def extract_hour(x):
    h, _ = map(int, x.split(":"))
    return h


def parse_data(x):
    # extract the last value for each attribute
    # 转换为字典
    # 我觉得输入的就是一个tuple
    x = x.set_index("Parameter").to_dict()["Value"]

    values = []

    for attr in attributes:
        if x.__contains__(attr):
            values.append(x[attr])
        else:
            values.append(np.nan)
    # 其实返回的是一个数组，如果x里面有attr那就填上，否则就是nan
    return values


def parse_id(id_, missing_ratio=0.1):
    data = pd.read_csv("./data/plrx/plrxA/{}.txt".format(id_))
    # set hour
    data["Time"] = data["Time"].apply(lambda x: extract_hour(x))

    # create data for 48 hours x 35 attributes
    observed_values = []
    # 一个txt里面有48个hour
    for h in range(1): #注定只有0
        observed_values.append(parse_data(data[data["Time"] == h])) #每次只针对一个h
    observed_values = np.array(observed_values)
    observed_masks = ~np.isnan(observed_values)

    # randomly set some percentage as ground-truth
    masks = observed_masks.reshape(-1).copy()
    obs_indices = np.where(masks)[0].tolist()
    miss_indices = np.random.choice(
        obs_indices, (int)(len(obs_indices) * missing_ratio), replace=False
    )
    masks[miss_indices] = False
    gt_masks = masks.reshape(observed_masks.shape)

    observed_values = np.nan_to_num(observed_values)
    observed_masks = observed_masks.astype("float32")
    gt_masks = gt_masks.astype("float32")


    # observed_masks 表示 observed_values 中，那些实际观测到的部分元素对应的位置为 1，缺失值对应的位置为 0；
    # gt_masks 我理解是那些随机生成的缺失位置
    # 这些矩阵可以用于训练和测试缺失数据的填充模型，以评估模型在缺失数据恢复方面的性能。
    return observed_values, observed_masks, gt_masks


def get_idlist():
    patient_id = []
    for filename in os.listdir("./data/plrx/plrxA"):
        match = re.search("\d{6}", filename) #尝试提取6位数据，也就是把所有txt前面的文件名提取出来
        if match:
            patient_id.append(match.group())
    patient_id = np.sort(patient_id)
    return patient_id


class croprec_Dataset(Dataset):
    def __init__(self, eval_length=1, use_index_list=None, missing_ratio=0.0, seed=0,is_TestLoader=False):
        self.eval_length = eval_length
        np.random.seed(seed)  # seed for ground truth choice

        self.observed_values = []
        self.observed_masks = []
        self.gt_masks = []

        idlist = get_idlist() #拿到所有需要的txt文件
        for id_ in idlist:
            try:
                observed_values, observed_masks, gt_masks = parse_id(
                    id_, missing_ratio
                )
                self.observed_values.append(observed_values)
                self.observed_masks.append(observed_masks)
                self.gt_masks.append(gt_masks)
            except Exception as e:
                print(id_, e)
                continue
        self.observed_values = np.array(self.observed_values)
        self.observed_masks = np.array(self.observed_masks)
        self.gt_masks = np.array(self.gt_masks)
        self.gt_masks = getGTmask(self.gt_masks)
        # if is_TestLoader:
        #     ov = self.observed_values.tolist()
        #     print(ov)
        #     gt = self.gt_masks.tolist()
        #     print(gt)

        # calc mean and std and normalize values
        # (it is the same normalization as Cao et al. (2018) (https://github.com/caow13/BRITS))
        tmp_values = self.observed_values.reshape(-1, 12)
        tmp_masks = self.observed_masks.reshape(-1, 12)
        mean = np.zeros(12)
        std = np.zeros(12)
        for k in range(12):
            c_data = tmp_values[:, k][tmp_masks[:, k] == 1]
            mean[k] = c_data.mean()
            std[k] = c_data.std()
        self.observed_values = (
            (self.observed_values - mean) / std * self.observed_masks
        )

                
        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=None, batch_size=16, missing_ratio=0.1):

    # only to obtain total length of dataset
    dataset = croprec_Dataset(missing_ratio=missing_ratio, seed=seed)
    # print(type(dataset))
    indlist = np.arange(len(dataset))
    # print(len(dataset))

    # 更改seed在这里
    np.random.seed(seed)
    np.random.shuffle(indlist)

    # 5-fold test
    start = (int)(nfold * 0.2 * len(dataset))
    end = (int)((nfold + 1) * 0.2 * len(dataset))
    test_index = indlist[start:end]
    remain_index = np.delete(indlist, np.arange(start, end))

    np.random.seed(seed)
    np.random.shuffle(remain_index)
    num_train = (int)(len(dataset) * 0.7)
    train_index = remain_index[:num_train]
    valid_index = remain_index[num_train:]
    
    train_index,valid_index,test_index = getLoaderIndex()
    print(test_index)
    print(train_index)
    print(valid_index)

    dataset = croprec_Dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=1)
    valid_dataset = croprec_Dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)
    test_dataset = croprec_Dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed,is_TestLoader=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)
    # sys.exit(1)
    # print(type(train_loader))
    # print(type(valid_loader))
    # print(type(test_loader))
    return train_loader, valid_loader, test_loader
