import numpy as np
import math
import pandas as pd
import csv
from numpy.linalg import inv
from sklearn.datasets import load_boston
import statsmodels.api as sm

train_data = np.load('EPA_14_training_set.npy')
test_data = np.load('EPA_14_testing_set.npy')

row = train_data.shape[0]
column = train_data.shape[1]

feature = ['AMB_TEMP','CO','NO','NO2','NOx','PM10','PM2.5','RAINFALL','RH','SO2']


# 求中山測站的預測
location = 17

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
    # 一個月的資料有20天，取連續33小時的data可以有448(480-32)筆
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



train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)


w = np.zeros(len(train_x[0]))
l_rate = 0.5
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



filename = "predict_24hour_板橋舊.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","predict value","real value","loss","percentage"])
for i in range(len(ans)):
    s.writerow(ans[i])
    
text.close()
