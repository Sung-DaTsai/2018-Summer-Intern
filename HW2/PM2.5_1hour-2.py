# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 15:41:09 2018

@author: roor
"""

import numpy as np
import math
import pandas
import csv
from numpy.linalg import inv


train_data = np.load('EPA_14_training_set.npy')
test_data = np.load('EPA_14_testing_set.npy')

row = train_data.shape[0]
column = train_data.shape[1]



# 求中山測站的預測
location = 1

data = []

for i in range(10):
    data.append([])

for j in range(location,row,18):
    for i in range(2,12):
        data[i%10-2].append(float(train_data[j][i]))



train_x = []
train_y = []

# 每 12 個月
for i in range(12):
    # 一個月的資料有20天，取連續10小時的data可以有471(480-9)筆
    # EX:1~10，2~11，3~12...471~480
    for j in range(471):
        train_x.append([])
        # 10種污染物
        for t in range(10):
            # 連續9小時
            for s in range(9):
                train_x[471*i+j].append(data[t][480*i+j+s] )
        train_y.append(data[6][480*i+j+9])
train_x = np.array(train_x)
train_y = np.array(train_y)

train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)


w = np.zeros(len(train_x[0]))
l_rate = 1.8
repeat = 10000

x_t = train_x.transpose()
s_gra = np.zeros(len(train_x[0]))



for i in range(repeat):
    hypo = np.dot(train_x,w)
    loss = hypo - train_y
    cost = np.sum(loss**2) / len(train_x)
    cost_a = math.sqrt(cost)
    gra = np.dot(x_t,loss)
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    if(i % 500==0):
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))


# save model
np.save('model1.npy',w)



#------------------------------------------------
# read model
w = np.load('model1.npy')


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



for i in range(2991-99):
    test_xx.append([])
    test_yy.append([])


for j in range(10):
    for k in range(10):
        for i in range(255):
            if k!=9:
                test_xx[i].append(data2[i+k][j])
            else:
                test_yy[i].append(data2[i+k][j])
        for i in range(264,447):
            if k!=9:
                test_xx[i-9].append(data2[i+k][j])
            else:
                test_yy[i-9].append(data2[i+k][j])
        for i in range(456,711):
            if k!=9:
                test_xx[i-18].append(data2[i+k][j])
            else:
                test_yy[i-18].append(data2[i+k][j])
        for i in range(720,951):
            if k!=9:
                test_xx[i-27].append(data2[i+k][j])
            else:
                test_yy[i-27].append(data2[i+k][j])
        for i in range(960,1215):
            if k!=9:
                test_xx[i-36].append(data2[i+k][j])
            else:
                test_yy[i-36].append(data2[i+k][j])
        for i in range(1224,1455):
            if k!=9:
                test_xx[i-45].append(data2[i+k][j])
            else:
                test_yy[i-45].append(data2[i+k][j])
        for i in range(1464,1719):
            if k!=9:
                test_xx[i-54].append(data2[i+k][j])
            else:
                test_yy[i-54].append(data2[i+k][j])
        for i in range(1728,1983):
            if k!=9:
                test_xx[i-63].append(data2[i+k][j])
            else:
                test_yy[i-63].append(data2[i+k][j])
        for i in range(1992,2223):
            if k!=9:
                test_xx[i-72].append(data2[i+k][j])
            else:
                test_yy[i-72].append(data2[i+k][j])
        for i in range(2232,2487):
            if k!=9:
                test_xx[i-81].append(data2[i+k][j])
            else:
                test_yy[i-81].append(data2[i+k][j])
        for i in range(2496,2727):
            if k!=9:
                test_xx[i-90].append(data2[i+k][j])
            else:
                test_yy[i-90].append(data2[i+k][j])
        for i in range(2736,2991):
            if k!=9:
                test_xx[i-99].append(data2[i+k][j])
            else:
                test_yy[i-99].append(data2[i+k][j])


test_x = np.array(test_xx)
test_y = np.array(test_yy)


test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)



ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)



total_loss = 0

for i in range(len(test_x)):
    ans[i].append(test_y[i][6])
    ans[i].append((ans[i][1] - test_y[i][6])**2)
    ans[i].append((ans[i][1] - test_y[i][6])/test_y[i][6])
    total_loss += ans[i][3]

avg_cost = math.sqrt(total_loss/len(test_x))
print("test_data Cost:", avg_cost)



filename = "predict_1hour_中山.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","predict value","real value","loss","percentage"])
for i in range(len(ans)):
    s.writerow(ans[i])
    
text.close()
