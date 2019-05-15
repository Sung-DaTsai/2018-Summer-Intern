import numpy as np
import math
import pandas as pd
import csv
from numpy.linalg import inv
from sklearn.datasets import load_boston
import statsmodels.api as sm

train_data = np.load('EPA_14_training_set.npy')
test_data = np.load('EPA_14_testing_set.npy')

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


# 求中山測站的預測
location = 1

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

#np.delete(predict, 0, 0)
for i in range(10):
    subdata.rename(columns={subdata.columns[i]: feature[i] }, inplace=True)



def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.01, 
                       threshold_out = 0.05, 
                       verbose=True):
    """ Perform a forward-backward feature selection 
    based on p-value from statsmodels.api.OLS
    Arguments:
        X - pandas.DataFrame with candidate features
        y - list-like with the target
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
           # if verbose:
            #    print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
           # if verbose:
            #    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included

result = stepwise_selection(subdata, predict)

print('resulting features:')
print(result)

'''
filename = "X.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
X = np.array(X)
for i in range(len(X)):
    s.writerow(X[i]) 
'''
# 開始train
data = np.transpose(data)
np.delete(data, 7, 0)
np.delete(data, 4, 0)

train_x = []
train_y = []

# 每 12 個月
for i in range(12):
    # 一個月的資料有20天，取連續10小時的data可以有471(480-9)筆
    # EX:1~10，2~11，3~12...471~480
    for j in range(471):
        train_x.append([])
        # 10種污染物
        # 但去掉不要的兩種汙染物('NOx','RAINFALL')
        for t in range(8):
            # 連續9小時
            for s in range(9):
                train_x[471*i+j].append(data[t][480*i+j+s])
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

np.delete(data2, 7, 1)
np.delete(data2, 4, 1)

test_xx = []
test_yy = []


data2 = data2.transpose()



for i in range(2991-99):
    test_xx.append([])
    test_yy.append([])


for j in range(8):
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



filename = "predict_1hour_Cost.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","predict value","real value","loss","percentage"])
for i in range(len(ans)):
    s.writerow(ans[i])
    
text.close()
