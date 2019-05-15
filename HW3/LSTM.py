from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import math
import csv
import time
from keras.layers import Input, Dense, Activation, Dropout, TimeDistributed, Flatten, Bidirectional, RepeatVector
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers.recurrent import LSTM
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras import regularizers


# load data
train_data2 = np.load('EPA_14_training_set.npy')
test_data2 = np.load('EPA_14_testing_set.npy')
row = train_data2.shape[0]

data3 = []

for i in range(10):
    data3.append([])


for location in range(18):
    for i in range(2,12):
        for j in range(location,row,18):
            data3[(i-2)%row].append(float(train_data2[j][i]))

data3 = np.array(data3)
data3 = np.transpose(data3)

row = data3.shape[0]
data3_time = []

for i in range(row):
    data3_time.append([])

for i in range(row):
    # add hour information
    data3_time[i].append(np.cos(i%24*2*np.pi/24))
    data3_time[i].append(np.sin(i%24*2*np.pi/24))
    # add month information
    data3_time[i].append(np.cos(i%5760//480*np.pi/6))
    data3_time[i].append(np.sin(i%5760//480*np.pi/6))


data3_time = np.array(data3_time)

data3 = np.concatenate((data3, data3_time), axis=1)

train_x2 = []

for i in range(int(data3.shape[0]/18)):
    train_x2.append([])

for i in range(int(data3.shape[0]/18)):
    for j in range(18):
        for k in range(14):
            train_x2[i].append(data3[i+5760*j,k])

train_x2 = np.array(train_x2)



data4 = []

for i in range(10):
    data4.append([])

for location in range(18):
    for j in range(location, test_data2.shape[0], 18):
        for i in range(2,12):
            data4[i%10-2].append(float(test_data2[j][i]))

data4 = np.array(data4)
data4 = np.transpose(data4)

row = data4.shape[0]

data4_time = []

for i in range(row):
    data4_time.append([])

for i in range(row):
    data4_time[i].append(np.cos(i%24*2*np.pi/24))
    data4_time[i].append(np.sin(i%24*2*np.pi/24))
    data4_time[i].append(np.cos(i%3000//250*np.pi/6))
    data4_time[i].append(np.sin(i%3000//250*np.pi/6))
        
        
data4_time = np.array(data4_time)

data4 = np.concatenate((data4, data4_time), axis=1)



test_xx2 = []

for i in range(int(data4.shape[0]/18)):
    test_xx2.append([])

for i in range(int(data4.shape[0]/18)):
    for j in range(18):    
        for k in range(14):
            test_xx2[i].append(data4[i+int(data4.shape[0]/18)*j,k])

test_xx2 = np.array(test_xx2)

train_x3 = train_x2


train_y = []

for i in range(12):
    if(i==0):
        train_xx = train_x3[:480-6,:]
        train_yy = train_x3[6:480,:]
    else:
        subtrain_xx = train_x3[480+480*(i-1):480+480*i-6,:]
        train_xx = np.concatenate((train_xx, subtrain_xx), axis=0)
        subtrain_y = train_x3[480+480*(i-1)+6:480+480*i,:]
        train_yy = np.concatenate((train_yy, subtrain_y), axis=0)

train_x = np.array(train_xx)


train_y = np.zeros((train_x.shape[0],18))       
counter = 0

for i in range(6,252,14):
    train_y[:,counter] = train_yy[:,i]
    counter += 1


test_xx3 = test_xx2

test_datay = np.zeros((3000,18))    
counter = 0
for i in range(6,252,14):
    test_datay[:,counter] = test_xx3[:,i]
    counter += 1


test_x = []
test_y = []

for i in range(2991-99-60):
    test_x.append([])
    test_y.append([])



test_x = test_xx3[:258,:]
test_x = np.concatenate((test_x, test_xx3[264:450,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[456:714,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[720:954,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[960:1218,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[1224:1458,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[1464:1722,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[1728:1986,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[1992:2226,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[2232:2490,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[2496:2730,:]), axis=0)
test_x = np.concatenate((test_x, test_xx3[2736:2994,:]), axis=0)

test_y = test_datay[6:264,:]
test_y = np.concatenate((test_y, test_datay[270:456,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[462:720,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[726:960,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[966:1224,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[1230:1464,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[1470:1728,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[1734:1992,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[1998:2232,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[2238:2496,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[2502:2736,:]), axis=0)
test_y = np.concatenate((test_y, test_datay[2742:3000,:]), axis=0)



# add rainfall information into NOx
for i in range(train_x.shape[0]):
    for j in range(7,252,14):
        if((train_x[i,j]>0)&(train_x[i,j]<10)):
            train_x[i,j-3] -= train_x[i,j]*10
            if(train_x[i,j-3]<0):
                train_x[i,j-3] = 0.1
        if(train_x[i,j]>=10):
            #train_x[i,j-3] /= train_x[i,j]
            train_x[i,j-3] = 10/train_x[i,j]
            if(train_x[i,j-3] - train_x[i,j]*10<0):
                train_x[i,j-3] = 0  
                       
for i in range(test_x.shape[0]):
    for j in range(7,252,14):
        if((test_x[i,j]>0)&(test_x[i,j]<10)):
            test_x[i,j-3] -= test_x[i,j]*10
            if(test_x[i,j-3]<0):
                test_x[i,j-3] = 0.1
        if(test_x[i,j]>=10):
            test_x[i,j-3] = 10/test_x[i,j]
            if(test_x[i,j-3] - test_x[i,j]*10<0):
                test_x[i,j-3] = 0  


# attribute: NOx(add rainfall information)、PM2.5、hours、months
train_xtemp = train_x[:,4:7]
train_xtemp = np.delete(train_xtemp,train_xtemp.shape[1]-2,1)

test_xtemp = test_x[:,4:7]
test_xtemp = np.delete(test_xtemp,test_xtemp.shape[1]-2,1)


for i in range(1,18):
    train_xtemp = np.concatenate((train_xtemp, train_x[:,4+14*i:7+14*i]), axis=1)
    train_xtemp = np.delete(train_xtemp,train_xtemp.shape[1]-2,1)


    test_xtemp = np.concatenate((test_xtemp, test_x[:,4+14*i:7+14*i]), axis=1)
    test_xtemp = np.delete(test_xtemp,test_xtemp.shape[1]-2,1)

train_x5 = train_xtemp
test_x5 = test_xtemp


for i in range(0,9):
    train_x5 = np.concatenate((train_x5, train_x[:,10+14*i:14+14*i]), axis=1)
    test_x5 = np.concatenate((test_x5, test_x[:,10+14*i:14+14*i]), axis=1)
                
                     
# do preprocessing transformation
sc = StandardScaler()
train_x5 = sc.fit_transform(train_x5)
test_x5 = sc.transform(test_x5)


train_x = train_x5
test_x = test_x5

#------------------------------------------------------------------------------
# Start to train
#------------------------------------------------------------------------------

# autoencoder model

# do dimension reduction

input_img = Input(shape=(72,))
encoded = Dense(70, activation='relu')(input_img)
encoder_output = Dense(50, activation='relu')(encoded)

decoded = Dense(70, activation='relu')(encoder_output)
decoded = Dense(72)(decoded)

autoencoder = Model(input_img, decoded)

encoder = Model(input=input_img, output=encoder_output)

encoded_input = Input(shape=(72,))
autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse','acc'])


print(train_x.shape)
print(test_x.shape)


history = autoencoder.fit(train_x, train_x, epochs=400, batch_size=100, shuffle=True, validation_split=0.1)
score1 = autoencoder.evaluate(test_x, test_x)

print("Test mse:",score1[1])
print("Test accuracy:",score1[2])
print(history.history.keys())


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("AE model mse")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(['train','validation'])
plt.savefig('AE model mse.png')
plt.show()

train_x_1 = encoder.predict(train_x)
test_x_1 = encoder.predict(test_x)

print(train_x_1.shape)

#------------------------------------------------------------------------------
# predict PM2.5

train_x = np.reshape(train_x_1, (train_x_1.shape[0],1,-1))
test_x = np.reshape(test_x_1, (test_x_1.shape[0],1,-1))

# 這裡用LSTM的model
model = Sequential()
model.add(LSTM(units=35, input_shape=(train_x.shape[1], train_x.shape[2]), dropout=0.1, activation='relu', return_sequences=False))
model.add(Dense(units=18))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','acc'])

#callback = EarlyStopping(monitor="val_loss", patience=10, verbose=1, mode="auto")

history = model.fit(train_x, train_y, batch_size=100, epochs=300, validation_split=0.2)
score = model.evaluate(test_x, test_y)
print("Test mse:", score[1])
print("Test acc:", score[2])
for layer in model.layers:
    weights = layer.get_weights()

a = model.predict(test_x)
ans = []


total_loss = 0

for i in range(test_y.shape[0]):
    for j in range(test_y.shape[1]):
        ans.append(["id_"+str(j+i*test_y.shape[1])])
        ans[j+i*test_y.shape[1]].append(a[i,j])


for i in range(test_y.shape[0]):
    for j in range(test_y.shape[1]):
        ans[j+i*test_y.shape[1]].append(test_y[i,j])
        ans[j+i*test_y.shape[1]].append((ans[j+i*test_y.shape[1]][1] - test_y[i,j])**2)
        ans[j+i*test_y.shape[1]].append((ans[j+i*test_y.shape[1]][1] - test_y[i,j])/test_y[i,j])
        total_loss += ans[j+i*test_y.shape[1]][3]


avg_cost = math.sqrt(total_loss/(test_y.shape[0]*test_y.shape[1]))
print("test_data Cost:", avg_cost)



filename = "predict_6hour_all_stations_autoencoder_by_both_data.csv"
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","predict value","real value","loss","percentage"])
for i in range(len(ans)):
    s.writerow(ans[i])
    
text.close()


plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("FC model accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['train','validation'])
plt.show()


plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("FC model mse")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend(['train','validation'])
plt.savefig('FC model mse.png')
plt.show()
