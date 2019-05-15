# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 10:53:25 2018

@author: roor
"""

import numpy as np
import math
import pandas as pd
import csv
from numpy.linalg import inv
from sklearn.ensemble import RandomForestRegressor


regr = RandomForestRegressor(n_estimators=100, max_depth=7)

# 求中山測站的預測
location = 15

train_data = np.load('EPA_14_training_set.npy')
test_data = np.load('EPA_14_testing_set.npy')
row = train_data.shape[0]



data = []

for i in range(10):
    data.append([])

for j in range(location,row,18):
    for i in range(2,12):
        data[(i-2)%10].append(float(train_data[j][i]))


# 開始train

train_x = []
train_y = []

# 每 12 個月
for i in range(12):
    # 一個月的資料有20天，取連續15小時的data可以有466(480-14)筆
    # EX:1~15，2~16，...466~480
    for j in range(448):
        train_x.append([])
        # 10種污染物
        for t in range(10):
            # 連續9小時
            for s in range(9):
                if(t!=6):
                    train_x[448*i+j].append(data[t][480*i+j+s])
        train_y.append(data[6][480*i+j+32])
train_x = np.array(train_x)
train_y = np.array(train_y)


data2 = []

for i in range(10):
    data2.append([])

for j in range(location, test_data.shape[0], 18):
    for i in range(2,12):
        data2[i%10-2].append(float(test_data[j][i]))

data2 = np.array(data2)

test_xx = []
test_yy = []


data2 = data2.transpose()

for i in range(2991-99-276):
    test_xx.append([])
    test_yy.append([])


for j in range(10):
    for k in range(10):
        for i in range(255-23):
            if((k!=9)&(j!=6)):
                test_xx[i].append(data2[i+k][j])
            elif(k==9):
                test_yy[i].append(data2[i+k+23][j])
        for i in range(264,447-23):
            if((k!=9)&(j!=6)):
                test_xx[i-9-23].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-9-23].append(data2[i+k+23][j])
        for i in range(456,711-23):
            if((k!=9)&(j!=6)):
                test_xx[i-18-46].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-18-46].append(data2[i+k+23][j])
        for i in range(720,951-23):
            if((k!=9)&(j!=6)):
                test_xx[i-27-69].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-27-69].append(data2[i+k+23][j])
        for i in range(960,1215-23):
            if((k!=9)&(j!=6)):
                test_xx[i-36-92].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-36-92].append(data2[i+k+23][j])
        for i in range(1224,1455-23):
            if((k!=9)&(j!=6)):
                test_xx[i-45-115].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-45-115].append(data2[i+k+23][j])
        for i in range(1464,1719-23):
            if((k!=9)&(j!=6)):
                test_xx[i-54-138].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-54-138].append(data2[i+k+23][j])
        for i in range(1728,1983-23):
            if((k!=9)&(j!=6)):
                test_xx[i-63-161].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-63-161].append(data2[i+k+23][j])
        for i in range(1992,2223-23):
            if((k!=9)&(j!=6)):
                test_xx[i-72-184].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-72-184].append(data2[i+k+23][j])
        for i in range(2232,2487-23):
            if((k!=9)&(j!=6)):
                test_xx[i-81-207].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-81-207].append(data2[i+k+23][j])
        for i in range(2496,2727-23):
            if((k!=9)&(j!=6)):
                test_xx[i-90-230].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-90-230].append(data2[i+k+23][j])
        for i in range(2736,2991-23):
            if((k!=9)&(j!=6)):
                test_xx[i-99-253].append(data2[i+k][j])
            elif(k==9):
                test_yy[i-99-253].append(data2[i+k+23][j])




test_x = np.array(test_xx)
test_y = np.array(test_yy)

regr.fit(train_x, train_y)
print(regr.feature_importances_)


a = regr.predict(test_x)


ans = []
for i in range(len(a)):
    ans.append(["id_"+str(i)])
    ans[i].append(a[i])



total_loss = 0

for i in range(len(a)):
    ans[i].append(test_y[i][6])
    ans[i].append((ans[i][1] - test_y[i][6])**2)
    ans[i].append((ans[i][1] - test_y[i][6])/test_y[i][6])
    total_loss += ans[i][3]

avg_cost = math.sqrt(total_loss/len(a))
print("test_data Cost:", avg_cost)



filename = "predict_24hour_陽明RF.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","predict value","real value","loss","percentage"])
for i in range(len(ans)):
    s.writerow(ans[i])
    
text.close()
