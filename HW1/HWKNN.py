"""
Created on Wed Jul 18 09:59:12 2018

@author: roor
"""

import numpy as np
import csv
from numpy.linalg import inv
import pandas as pd 
from numpy import *
import matplotlib.pyplot as plt
import math
from sklearn.neighbors import KNeighborsRegressor
import keras

csv_data = pd.read_csv('EPA_data.csv') 
print(csv_data.shape) 
csv_data = np.array(csv_data)
print(type(csv_data))
location = np.array([[12,17],[13,20],[17,20],[21,13],[9,19],[13,19],[22,21],[16,11],[14,25],[18,13],[12,4],[18,19],[13,32],[3,12],[13,16],[15,18],[2,36],[1,20]])
weight = np.zeros((30,38,18))




data = []
for i in range(16):
    data.append([])

for i in range(16):
    for j in range(csv_data.shape[0]):
        data[i].append(csv_data[j][i+1])


# 將檔案讀進去之後進行處理，nan值改成-0.0001，方便後面運算
data = np.array(data)
where = np.isnan(data)
data[where] = -0.0001



data = np.transpose(data)

# 將檔案讀進去之後進行處理，刪去不要的feature
data = np.delete(data, 12, 1)
data = np.delete(data, 8, 1)
data = np.delete(data, 3, 1)
data = np.delete(data, 1, 1)


final_data = np.zeros((int(data.shape[0]/18),12,30,38))

div_data = []

# 最外層先擺18個測站
for i in range(18):
    div_data.append([])
# 第二層擺每個測站的所有時間
    for j in range(int(data.shape[0]/18)):
        div_data[i].append([])
# 第三層擺12種汙染物
        for k in range(12):
            div_data[i][j].append([])

total = 0
counter = 0


# 將nan值改成-0.0001後，此處便是將這些空缺的值補入該時間點該feature的平均值
# 設成-0.0001可以與任何有實際數據的部分做區隔
for i in range(18):
    for j in range(int(data.shape[0]/18)):
        for k in range(12):
            div_data[i][j][k] = data[i+18*j][k]
            # 針對nan值(-0.0001)進行填補
            if(data[i+18*j][k] == -0.0001):
                for ave in range(18):
                    # 計算平均時不能將-0.0001也納入考量
                    if(data[ave+18*j][k] != -0.0001):
                        total += data[ave+18*j][k]
                        counter += 1
                div_data[i][j][k] = total/counter
                total = 0
                counter = 0

place = np.zeros((1140,2))
for i in range(30):
    for j in range(38):
        place[i*38+j,0] = i
        place[i*38+j,1] = j


# 將有測站的數據補完後填入final_data，即為四維陣列的形式
for i in range(18):
    for j in range(int(data.shape[0]/18)):
        for k in range(12):
            final_data[j,k,location[i,0],location[i,1]] = div_data[i][j][k]


div_data = np.array(div_data)

final_data = np.reshape(final_data, (int(data.shape[0]/18),12,30*38))

for i in range(18):
    for j in range(50):
        for k in range(12):
            neigh = KNeighborsRegressor(n_neighbors=4)
            neigh.fit(location, div_data[:,j,k])
            final_data[j,k,:] = neigh.predict(place)


final_data = np.reshape(final_data, (int(data.shape[0]/18),12,30,38))

keras.layers.ConvLSTM2D(3, 3, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, go_backwards=False, stateful=False, dropout=0.0, recurrent_dropout=0.0)

csv_data2 = pd.read_csv('grid_location.csv') 
csv_data2 = np.array(csv_data2)


file_path1 = './grid_location.csv'
df_grid = pd.read_csv(file_path1,encoding = 'utf-8').drop(['Unnamed: 0'], 1)
df_grid.head()

file_path2 = './EPA_data.csv'
df14_air = pd.read_csv(file_path2,encoding='utf-8').drop(['Unnamed: 0'], 1)
df14_air.head()

file_path3 = './grid_version_op.npy'
arr = np.load(file_path3)

print(arr.shape)

plt.imshow(arr[0][0])
plt.show()
plt.imshow(final_data[0][0])
plt.savefig('final_data[0][0].png')
plt.show()