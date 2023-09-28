
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score

def create_local_learner(data_list, label_list, num_clients=8, initial='clients'):
     #create a list of client names
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    #randomize the data
    data = list(zip(data_list, label_list))
    random.shuffle(data)  
    #shard data and place at each client
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    assert(len(shards) == len(client_names))
    
    return {client_names[i] : shards[i] for i in range(len(client_names))} 


def batch_data(data_shard, bs=32):
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)


def weight_scalling_factor(clients_trn_data, client_name):
    client_names = list(clients_trn_data.keys())
    bs = list(clients_trn_data[client_name])[0][0].shape[0]
    #first calculate the total training data points across clinets
    global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
    # get the total number of data points held by a client
    local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
    if debug:
        print('global_count', global_count, 'local_count', local_count, 'bs', bs)
    return local_count/global_count


def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    #get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad

def test_model(X_test, Y_test,  model, comm_round):
    
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    y_predicted = model.predict(X_test)
    loss = cce(Y_test, y_predicted)
    acc = accuracy_score(tf.argmax(y_predicted, axis=1), tf.argmax(Y_test, axis=1))
    print('Comm_round: {} | Global_acc: {:.3%} | Global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss