import numpy as np
import pandas as pd
import random
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from imutils import paths
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt    
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense

debug = 0

train_data=pd.read_csv('mitbih_train.csv',header=None)
test_data=pd.read_csv('mitbih_test.csv',header=None)
target_train=train_data.iloc[:,-1]
target_test=test_data.iloc[:,-1]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=train_data.iloc[:,:-1].values
X_test=test_data.iloc[:,:-1].values

#noisy data for robust model training 
X_train_noise=np.array(X_train)
X_test_noise=np.array(X_test)

plt.plot(X_train[16])

def gaussian_noise(signal):
    noise=np.random.normal(0,0.01,187)
    out=signal+noise
    return out

def make_it_noisy(noisy_train,noisy_test): 
    for i in range(len(noisy_train)):
        X_train_noise[i,:187]= gaussian_noise(noisy_train[i,:187])
    for i in range(len(noisy_test)):
        X_test_noise[i,:187]= gaussian_noise(noisy_test[i,:187])
        
    X_train_noise= X_train_noise.reshape(len(X_train_noise), X_train_noise.shape[1],1)
    X_train= X_train.reshape(len(X_train), X_train.shape[1],1)
    X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
    X_test_noise= X_test_noise.reshape(len(X_test_noise), X_test_noise.shape[1],1)
    X_train=np.float32(X_train)
    X_train_noise=np.float32(X_train_noise)
    X_test=np.float32(X_test)
    X_test_noise=np.float32(X_test_noise)
    return X_train_noise,X_test_noise
make_it_noisy(X_train_noise,X_test_noise)

smote = SMOTE(sampling_strategy='auto')  
X_resampled, y_resampled = smote.fit_resample(X_train,y_train) 

def oneD_CNN_model(input_shape,num_classes):
# Define the model
    model = keras.models.Sequential()

    # Add a 1D convolutional layer with 32 filters, kernel size 3, and ReLU activation
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)))
    # You need to specify the input_shape which is (sequence_length, input_dim), where input_dim is 1 for a single channel.

    # Add a max-pooling layer with pool size 2
    model.add(MaxPooling1D(pool_size=2))

    # Flatten the output of the convolutional layer
    model.add(Flatten())

    # Add one or more fully connected (dense) layers
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # The last layer should have as many units as the number of classes in your classification problem.

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print a summary of the model's architecture
    model.summary()
    return model
print(oneD_CNN_model(187,5))