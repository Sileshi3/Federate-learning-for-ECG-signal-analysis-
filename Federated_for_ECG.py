import numpy as np
import pandas as pd
import random
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer 
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from imutils import paths
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt    
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from imblearn.over_sampling import SMOTE
from collections import Counter 

from Federated_Model import *
from model import *
from plot import ploter
debug = 0

train_data=pd.read_csv('Dataset/mitbih_train.csv',header=None)
test_data=pd.read_csv('Dataset/mitbih_test.csv',header=None)
target_train=train_data.iloc[:,-1]
target_test=test_data.iloc[:,-1]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)

X_train=train_data.iloc[:,:-1].values
X_test=test_data.iloc[:,:-1].values

#noisy data for robust model training 
X_train_noise=np.array(X_train)
X_test_noise=np.array(X_test)

#plt.plot(X_train[16])

def gaussian_noise(signal):
    noise=np.random.normal(0,0.01)
    out=signal+noise
    return out

def make_it_noisy(noisy_train,noisy_test): 
    for i in range(len(noisy_train)):
        X_train_noise[i,:187]= gaussian_noise(noisy_train[i,:187])
    for i in range(len(noisy_test)):
        X_test_noise[i,:187]= gaussian_noise(noisy_test[i,:187])
    X_train_noisey= X_train_noise.reshape(len(X_train_noise), X_train_noise.shape[1],1)
    X_train= X_train.reshape(len(X_train), X_train.shape[1],1)
    X_test = X_test.reshape(len(X_test), X_test.shape[1],1)
    X_test_noisey= X_test_noise.reshape(len(X_test_noise), X_test_noise.shape[1],1)
    X_train=np.float32(X_train)
    X_train_noisey=np.float32(X_train_noise)
    X_test=np.float32(X_test)
    X_test_noise=np.float32(X_test_noise)
    return X_train_noisey,X_test_noisey 

smote = SMOTE(sampling_strategy='auto')  
X_resampled, y_resampled = smote.fit_resample(X_train,y_train) 

#To label a class name for confusion matrix
def class_name_labeler(data_portion):
    label_to_name = {0:'Norma Beat',1:'Supraventricular',2:'Ventricular',3:'Fusion',4:'Unknown Beat',}
    class_names=[]
    encoded_labels = []
    for i in range (len(data_portion)):
        encoded_labels.append(data_portion[i])
    for label in encoded_labels:
        class_name = label_to_name[label]
        class_names.append(class_name)
    return class_names

class_labels = target_test
class_distribution = class_labels.value_counts() 
label_name=class_name_labeler(class_distribution.index)

# Plot the class distribution
#plt.figure(figsize=(5,3))
#plt.bar(label_name, class_distribution.values)
#plt.xlabel('Heart Beat Type')
#plt.ylabel('Count')
#plt.title('Class Distribution')
#plt.xticks(rotation=45)
#plt.show()

# Calculate and print the class imbalance ratio
imbalance_ratio = class_distribution.min() / class_distribution.max()
print(f"Class Imbalance Ratio: {imbalance_ratio:.2f}")

# Upsampling the imbalance in the dataset
smote = SMOTE(sampling_strategy='auto')  
X_resampled, y_resampled = smote.fit_resample(X_train,y_train)  

#Creat local learners 
len(X_train), len(X_test), len(y_train), len(y_test) 
clients = create_local_learner(X_resampled, y_resampled, num_clients=10, initial='client')

#process and batch the training data for each client
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)
    
#process and batch the test set  
test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))


lr = 0.01
comms_round = 10
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr,
                                           decay=lr / comms_round,
                                           momentum=0.9)    

#initialize global model 
feature_size=187
smlp_global = SimpleMLP()
global_model = smlp_global.build(feature_size, 5)  
global_acc_list = []
global_loss_list = []

#Global training loop
for comm_round in range(comms_round): 
    global_weights = global_model.get_weights() 
    scaled_local_weight_list = list()
    all_client_names = list(clients_batched.keys())#randomize client data - using keys
    client_names = random.sample(all_client_names, k=10)
    #print(client_names, len(client_names))
    random.shuffle(client_names)
    for client in client_names:
        smlp_local = SimpleMLP()
        local_model = smlp_local.build(feature_size, 5)
        local_model.compile(loss=loss, 
                            optimizer=optimizer, 
                            metrics=metrics)
        
        local_model.set_weights(global_weights) #set local model weight to the weight of the global model
        local_model.fit(clients_batched[client], epochs=1, verbose=0)
        
        #scale the model weights and add to list
        scaling_factor = 0.1 # weight_scalling_factor(clients_batched, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        K.clear_session()
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    global_model.set_weights(average_weights) #update global model 

    for(X_test, Y_test) in test_batched:
        global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
        global_acc_list.append(global_acc)
        global_loss_list.append(global_loss)
 
#Evaluation using acc curve,error and confusion matrix        
ploter(global_acc_list,global_loss_list)
cm = confusion_matrix(target_test, y_pred)
target_names = list("NSVFQ") 
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)  # Adjust font size for better readability
sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix') 
clf_report = classification_report(target_test, y_pred, target_names=target_names, output_dict=True)
print(classification_report(target_test, y_pred, target_names=target_names))
plt.show()