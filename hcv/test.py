import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = np.random.random(size=(20, 2))
print(data)
stand_scaler = StandardScaler()
std_data = stand_scaler.fit_transform(data) # 标准化
np.random.shuffle(std_data)
origin_std_data = stand_scaler.inverse_transform(std_data) # 逆标准化
print(origin_std_data)
