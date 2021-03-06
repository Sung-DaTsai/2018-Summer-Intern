# -*- coding: utf-8 -*-
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


csv_data = pd.read_csv('EPA_data.csv') 
print(csv_data.shape) 
csv_data = np.array(csv_data)
print(type(csv_data))



data = []
for i in range(16):
    data.append([])

for i in range(16):
    for j in range(csv_data.shape[0]):
        data[i].append(csv_data[j][i+1])

data = np.array(data)
where = np.isnan(data)
data[where] = -0.0001



data = np.transpose(data)

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
for i in range(18):
    for j in range(int(data.shape[0]/18)):
        for k in range(12):
            div_data[i][j][k] = data[i+18*j][k]
            if(data[i+18*j][k] == -0.0001):
                for ave in range(18):
                    if(data[ave+18*j][k] != -0.0001):
                        total += data[ave+18*j][k]
                        counter += 1
                div_data[i][j][k] = total/counter
                total = 0
                counter = 0
                
            if(div_data[i][j][k] < 0):
                div_data[i][j][k] = 0
                

for i in range(18):
    for j in range(int(data.shape[0]/18)):
        for k in range(12):
            if(i==0):
                final_data[j][k][12][17] = div_data[i][j][k]
            elif(i==1):
                final_data[j][k][13][20] = div_data[i][j][k]
            elif(i==2):
                final_data[j][k][17][20] = div_data[i][j][k]
            elif(i==3):
                final_data[j][k][21][13] = div_data[i][j][k]
            elif(i==4):
                final_data[j][k][9][19] = div_data[i][j][k]
            elif(i==5):
                final_data[j][k][13][19] = div_data[i][j][k]
            elif(i==6):
                final_data[j][k][22][21] = div_data[i][j][k]
            elif(i==7):
                final_data[j][k][16][11] = div_data[i][j][k]
            elif(i==8):
                final_data[j][k][14][25] = div_data[i][j][k]
            elif(i==9):
                final_data[j][k][18][13] = div_data[i][j][k]    
            elif(i==10):
                final_data[j][k][12][4] = div_data[i][j][k]
            elif(i==11):
                final_data[j][k][18][19] = div_data[i][j][k]
            elif(i==12):
                final_data[j][k][13][32] = div_data[i][j][k]
            elif(i==13):
                final_data[j][k][3][12] = div_data[i][j][k]
            elif(i==14):
                final_data[j][k][13][16] = div_data[i][j][k]
            elif(i==15):
                final_data[j][k][15][18] = div_data[i][j][k]
            elif(i==16):
                final_data[j][k][2][36] = div_data[i][j][k]
            elif(i==17):
                final_data[j][k][1][20] = div_data[i][j][k]






for i in range(int(data.shape[0]/18)):
    for j in range(12):
        final_data[i,j,:,:] = (final_data[i,j,12,17]+final_data[i,j,13,20]+final_data[i,j,17,20]+final_data[i,j,21,13]+final_data[i,j,9,19]+final_data[i,j,13,19]+final_data[i,j,22,21]+final_data[i,j,16,11]+final_data[i,j,14,25]+final_data[i,j,18,13]+final_data[i,j,12,4]+final_data[i,j,18,19]+final_data[i,j,13,32]+final_data[i,j,3,12]+final_data[i,j,13,16]+final_data[i,j,15,18]+final_data[i,j,2,36]+final_data[i,j,1,20])/18

for i in range(18):
    for j in range(int(data.shape[0]/18)):
        for k in range(12):
            if(i==0):
                final_data[j][k][12][17] = div_data[i][j][k]
            elif(i==1):
                final_data[j][k][13][20] = div_data[i][j][k]
            elif(i==2):
                final_data[j][k][17][20] = div_data[i][j][k]
            elif(i==3):
                final_data[j][k][21][13] = div_data[i][j][k]
            elif(i==4):
                final_data[j][k][9][19] = div_data[i][j][k]
            elif(i==5):
                final_data[j][k][13][19] = div_data[i][j][k]
            elif(i==6):
                final_data[j][k][22][21] = div_data[i][j][k]
            elif(i==7):
                final_data[j][k][16][11] = div_data[i][j][k]
            elif(i==8):
                final_data[j][k][14][25] = div_data[i][j][k]
            elif(i==9):
                final_data[j][k][18][13] = div_data[i][j][k]    
            elif(i==10):
                final_data[j][k][12][4] = div_data[i][j][k]
            elif(i==11):
                final_data[j][k][18][19] = div_data[i][j][k]
            elif(i==12):
                final_data[j][k][13][32] = div_data[i][j][k]
            elif(i==13):
                final_data[j][k][3][12] = div_data[i][j][k]
            elif(i==14):
                final_data[j][k][13][16] = div_data[i][j][k]
            elif(i==15):
                final_data[j][k][15][18] = div_data[i][j][k]
            elif(i==16):
                final_data[j][k][2][36] = div_data[i][j][k]
            elif(i==17):
                final_data[j][k][1][20] = div_data[i][j][k]

for i in range(int(data.shape[0]/18)):
    for j in range(12):
        temp = final_data[i,j,12,17]
        final_data[i,j,11:14,16:19] = (final_data[i,j,12,17] + final_data[i,j,0,0])/2
        final_data[i,j,12,17] = temp
        
        temp = final_data[i,j,13,20]
        final_data[i,j,12:15,19:22] = (final_data[i,j,13,20] + final_data[i,j,0,0])/2
        final_data[i,j,13,20] = temp
        
        temp = final_data[i,j,17,20]
        final_data[i,j,16:19,19:22] = (final_data[i,j,17,20] + final_data[i,j,0,0])/2
        final_data[i,j,17,20] = temp
        
        temp = final_data[i,j,21,13]
        final_data[i,j,20:23,12:15] = (final_data[i,j,21,13] + final_data[i,j,0,0])/2
        final_data[i,j,21,13] = temp
        
        temp = final_data[i,j,9,19]
        final_data[i,j,8:11,18:21] = (final_data[i,j,9,19] + final_data[i,j,0,0])/2
        final_data[i,j,9,19] = temp
        
        temp = final_data[i,j,13,19]
        final_data[i,j,12:15,18:21] = (final_data[i,j,13,19] + final_data[i,j,0,0])/2
        final_data[i,j,13,19] = temp
        
        temp = final_data[i,j,22,21]
        final_data[i,j,21:24,20:23] = (final_data[i,j,22,21] + final_data[i,j,0,0])/2
        final_data[i,j,22,21] = temp
        
        temp = final_data[i,j,16,11]
        final_data[i,j,15:18,10:13] = (final_data[i,j,16,11] + final_data[i,j,0,0])/2
        final_data[i,j,16,11] = temp
        
        temp = final_data[i,j,14,25]
        final_data[i,j,13:16,24:27] = (final_data[i,j,14,25] + final_data[i,j,0,0])/2
        final_data[i,j,14,25] = temp
        
        temp = final_data[i,j,18,13]
        final_data[i,j,17:20,12:15] = (final_data[i,j,18,13] + final_data[i,j,0,0])/2
        final_data[i,j,18,13] = temp
        
        temp = final_data[i,j,12,4]
        final_data[i,j,11:14,3:6] = (final_data[i,j,12,4] + final_data[i,j,0,0])/2
        final_data[i,j,12,4] = temp
        
        temp = final_data[i,j,18,19]
        final_data[i,j,17:20,18:21] = (final_data[i,j,18,19] + final_data[i,j,0,0])/2
        final_data[i,j,18,19] = temp
        
        temp = final_data[i,j,13,32]
        final_data[i,j,12:15,31:34] = (final_data[i,j,13,32] + final_data[i,j,0,0])/2
        final_data[i,j,13,32] = temp
        
        temp = final_data[i,j,3,12]
        final_data[i,j,2:5,11:14] = (final_data[i,j,3,12] + final_data[i,j,0,0])/2
        final_data[i,j,3,12] = temp
        
        temp = final_data[i,j,13,16]
        final_data[i,j,12:15,15:18] = (final_data[i,j,13,16] + final_data[i,j,0,0])/2
        final_data[i,j,13,16] = temp
        
        temp = final_data[i,j,15,18]
        final_data[i,j,14:17,17:20] = (final_data[i,j,15,18] + final_data[i,j,0,0])/2
        final_data[i,j,15,18] = temp
        
        temp = final_data[i,j,2,36]
        final_data[i,j,1:4,35:38] = (final_data[i,j,2,36] + final_data[i,j,0,0])/2
        final_data[i,j,2,36] = temp
        
        temp = final_data[i,j,1,20]
        final_data[i,j,0:3,19:22] = (final_data[i,j,1,20] + final_data[i,j,0,0])/2
        final_data[i,j,1,20] = temp


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
plt.imshow(final_data[0][1])
plt.show()

