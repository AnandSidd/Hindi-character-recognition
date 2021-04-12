import numpy as np
import matplotlib.pyplot as plt
import os
from keras.models import *
import pandas as pd
from keras.layers import *
from keras.optimizers import *
import skimage.io as io
from sklearn.model_selection import train_test_split

train_dir = os.listdir('./Desktop/DevanagariHandwrittenCharacterDataset2/Train')
test_dir = os.listdir('./Desktop/DevanagariHandwrittenCharacterDataset2/Test')

def create_train_data():
    ref = {}
    X_train = []
    Y_train = []
    for i in range(0,10,1):
        p = 37+i
        ref[p] = i
    for names in train_dir:
        x = names.split('_')
        if x[0]=='character':
            ref[int(x[1])] = x[2]
        di = './Desktop/DevanagariHandwrittenCharacterDataset2/Train/'+names
        images = os.listdir(di)
        for files in images:
            files = di+'/'+files
            img = io.imread(files)
            X_train.append(img)
            if x[0]=='character':
                Y_train.append(int(x[1]))
    X_train = np.array(X_train)
    Y_train = np.asarray(Y_train)
    return X_train, Y_train, ref


def create_test_data():
    X_test = []
    Y_test = []
    for names in test_dir:
        x = names.split('_')
        di = './Desktop/DevanagariHandwrittenCharacterDataset2/Test/'+names
        images = os.listdir(di)
        for files in images:
            files = di+'/'+files
            img = io.imread(files)
            X_test.append(img)
            if x[0]=='character':
                Y_test.append(int(x[1]))
    X_test = np.array(X_test)
    Y_test = np.asarray(Y_test)
    return X_test, Y_test


X_train, Y_train, ref = create_train_data()
X_test, Y_test = create_test_data()

X_train = X_train.reshape([61200,-1])
print(X_train.shape)
df = pd.DataFrame(X_train)
df.to_csv('X_train.csv')

print(X_test.shape)
X_test = X_test.reshape([10800,1024])
print(X_test.shape)
df = pd.DataFrame(X_test)
df.to_csv('X_test.csv')


Y_train = Y_train.reshape([61200,-1])
print(Y_train.shape)
df = pd.DataFrame(Y_train)
df.to_csv('Y_train.csv')

Y_test = Y_test.reshape([10800,-1])
print(Y_test.shape)
df = pd.DataFrame(Y_test)
df.to_csv('Y_test.csv')

np.save('References.npy',ref)
