import os 
import tensorflow as tf 
import pandas as pd
from tensorflow.keras.utils import to_categorical 
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

num_clients=3
initial='local'
client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
print(client_names)

train_data=pd.read_csv('Dataset/mitbih_train.csv',header=None)
test_data=pd.read_csv('Dataset/mitbih_test.csv',header=None)
target_train=train_data.iloc[:,-1]
target_test=test_data.iloc[:,-1]
y_train=to_categorical(target_train)
y_test=to_categorical(target_test)
x_train=train_data.iloc[:,:-1].values
x_test=test_data.iloc[:,:-1].values 

def create_local(data_list, label_list, num_clients, initial='clients'): 
    local_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)] 
    data = list(zip(data_list, label_list))
    random.shuffle(data)   
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    assert(len(shards) == len(local_names))
    return {local_names[i] : shards[i] for i in range(len(local_names))} 

def batch_data(data_shard, bs=32): 
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

local_model = create_local(x_train, y_train, num_clients=3, initial='local')  
local_batched = dict()
for (local_name, data) in local_model.items():
    local_batched[local_name] = batch_data(data)  

all_client_names = list(local_batched.keys())#randomize client data - using keys
client_names = random.sample(all_client_names, k=3)
for clientc in client_names:
    print(clientc)

test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))