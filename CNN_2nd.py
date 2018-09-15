# -*- coding: utf-8 -*-
"""
@author: Wenxuan Xu 
email: rifflexiansen@qq.com
"""

from __future__ import print_function
import numpy as np
from sklearn.preprocessing import normalize
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
import theano

def DCDE_CNN(img_rows, img_cols, filter_size, kernel_size, pool_size, x_train, y_train, x_test, y_test):

    input_shape = (img_rows, img_cols)
    #Conv layer
    model = Sequential()
    model.add(Conv1D(filter_size, kernel_size=kernel_size,
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.01),
                     input_shape=input_shape))
    ##MaxPooling
    model.add(MaxPooling1D(pool_size=pool_size))
    #full connection
    model.add(Flatten())
    model.add(Dense(output_size,activation='relu'))
    model.add(Dropout(0.25))
    #softmax 
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    #============================================================================================
    get_feature = theano.function([model.layers[0].input],model.layers[3].output,allow_input_downcast=True)
    FC_train_feature = get_feature(x_train)
    FC_test_feature = get_feature(x_test)
    FC_train_feature = normalize(FC_train_feature, norm='l2')
    FC_test_feature = normalize(FC_test_feature, norm='l2')
    
    return (FC_train_feature, FC_test_feature)

#============================================================================================
kmer = 3
batch_size = 40
num_classes = 2
epochs = 1
filter_size = 64
kernel_size = 5
pool_size = 2
output_size = 128
#============================================================================================
x_train = np.load('x_train.npy') 
x_test = np.load('x_test.npy')  
y_train = np.load('y_train.npy')  
y_test = np.load('y_test.npy') 
index = np.load('index.npy')
img_cols, img_rows  = len(index), 251-kmer
y_train_new = y_train
y_test_new = y_test
y_train = keras.utils.to_categorical(y_train, 2)
y_test = keras.utils.to_categorical(y_test, 2)
np.save('y_train_new.npy',y_train_new)  
np.save('y_test_new.npy',y_test_new) 

#============================================================================================
#CNN
(FC_train_feature, FC_test_feature) = DCDE_CNN(img_rows, img_cols, filter_size, kernel_size, pool_size, x_train, y_train, x_test, y_test)

np.save('FC_train_feature.npy',FC_train_feature) 
np.save('FC_test_feature.npy',FC_test_feature)  




