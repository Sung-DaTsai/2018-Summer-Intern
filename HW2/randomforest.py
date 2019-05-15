import numpy as np
import math
import pandas as pd
import csv
from numpy.linalg import inv
from sklearn.datasets import load_boston
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(max_depth=3, random_state=0)

'''
regr.fit(X, y)
print(regr.feature_importances_)
print(regr.predict([[0, 0, 0, 0]]))
'''


train_data = np.load('EPA_14_training_set.npy')
test_data = np.load('EPA_14_testing_set.npy')


# 求中山測站的預測
location = 5


filename = "train_data.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['Time','Place','AMB_TEMP','CO','NO','NO2','NOx','PM10','PM2.5','RAINFALL','RH','SO2'])
train_data = np.array(train_data)
for i in range(len(train_data)):
    s.writerow(train_data[i]) 
    
    
filename = "test_data.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(['Time','Place','AMB_TEMP','CO','NO','NO2','NOx','PM10','PM2.5','RAINFALL','RH','SO2'])
test_data = np.array(test_data)
for i in range(len(test_data)):
    s.writerow(test_data[i]) 

row = train_data.shape[0]
column = train_data.shape[1]

feature = ['AMB_TEMP','CO','NO','NO2','NOx','PM10','PM2.5','RAINFALL','RH','SO2']





data = []

for i in range(10):
    data.append([])

for j in range(location,row,18):
    for i in range(2,12):
        data[(i-2)%10].append(float(train_data[j][i]))


data = np.transpose(data)
subdata = data[:5754,:]
predict = data[6:5760,6]
subdata = pd.DataFrame(subdata)


data2 = []

for i in range(10):
    data2.append([])

for j in range(location, test_data.shape[0], 18):
    for i in range(2,12):
        data2[(i-2)%10].append(float(test_data[j][i]))

data2 = np.array(data2)
data2 = np.transpose(data2)

subdata2 = data2[:2994,:]
predict2 = data2[6:3000,6]
subdata2 = pd.DataFrame(subdata2)



regr.fit(subdata, predict)
print(regr.feature_importances_)

a = regr.predict(subdata2)

ans = []
for i in range(len(a)):
    ans.append(["id_"+str(i)])
    ans[i].append(a[i])



total_loss = 0

for i in range(len(a)):
    ans[i].append(predict2[i])
    ans[i].append((ans[i][1] - predict2[i])**2)
    ans[i].append((ans[i][1] - predict2[i])/predict2[i])
    total_loss += ans[i][3]

avg_cost = math.sqrt(total_loss/len(a))
print("test_data Cost:", avg_cost)



filename = "predict_6hour_陽明RF.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","predict value","real value","loss","percentage"])
for i in range(len(ans)):
    s.writerow(ans[i])
    
text.close()
