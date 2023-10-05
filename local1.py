import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
import flwr as fl
import ssl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from data_distributor import *
from tensorflow.keras.utils import to_categorical
ssl._create_default_https_context = ssl._create_unverified_context
import random

input_shape=187
num_classes=5
lr = 0.01
comms_round = 10
loss='categorical_crossentropy'
metrics = ['accuracy']
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=lr,
                                           decay=lr / comms_round,
                                           momentum=0.9)  
class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
    
client_model = SimpleMLP()
client_model = client_model.build(input_shape, 5)
client_model.compile(loss=loss,
                    optimizer=optimizer, 
                    metrics=metrics) 

all_local_names = list(local_batched.keys())
local_names = random.sample(all_local_names,k=3) 
 


class FlowerClient(fl.client.NumPyClient):
    
    def get_parameters(self,config):
        return client_model.get_weights() 
    
    def fit(self, parameters, config):
        client_model.set_weights(parameters)
        fitted_model=client_model.fit(local_batched['local_1'],epochs=10,batch_size=32)
        print(("Fit History: ",fitted_model.history))
        stry=fitted_model.history 
        
        plt.figure(figsize=(6,3))
        plt.ylabel("Mean Absolute Error")
        plt.xlabel("Epoch") 
        plt.plot(list(range(0,len(stry['loss']))), stry['loss'],color='red') 
        plt.plot(list(range(0,len(stry['accuracy']))), stry['accuracy'],color='blue')
        plt.ylabel("Accuracy of Global Model")
        plt.xlabel("Epoch")   
        
        
        # fig, ax1 = plt.subplots() 
        # ax1.set_xlabel('Epochs')
        # ax1.set_ylabel('Loss and Accuracy')
        # ax1.plot(len(stry['loss']), stry['loss'], color='tab:blue', label='Loss')
        # ax1.plot(len(stry['accuracy']), stry['accuracy'], color='tab:red', label='Accuracy')
        # ax1.legend(loc='upper left') 
        # plt.title('Loss and Accuracy Over Epochs') 
        plt.show()
        return client_model.get_weights(),len(x_train),{}
    
    def evaluate(self, parameters, config):
        client_model.set_weights(parameters)
        loss, accuracy=client_model.evaluate(x_test,y_test)
        print("Evaluation Accuracy: ", accuracy)
        return loss, len(x_test),{"Accuracy": accuracy}
     
fl.client.start_numpy_client(server_address="127.0.0.1:8080",client=FlowerClient(),)



