import tensorflow as tf
import flwr as fl 
strategy=fl.server.strategy.FedAvg()
fl.server.start_server(strategy=strategy,
                       server_address="0.0.0.0:8080",
                       config=fl.server.ServerConfig(num_rounds=10))

# class SimpleMLP:
#     @staticmethod
#     def build(shape, classes):
#         model = Sequential()
#         model.add(Dense(200, input_shape=(shape,)))
#         model.add(Activation("relu"))
#         model.add(Dense(200))
#         model.add(Activation("relu"))
#         model.add(Dense(classes))
#         model.add(Activation("softmax"))
#         return model


# smlp_global = SimpleMLP()
# global_model = smlp_global.build(feature_size, 5)  
# global_acc_list = []
# global_loss_list = []

# for comm_round in range(comms_round): 
#     global_weights = global_model.get_weights() 
#     scaled_local_weight_list = list()
#     all_client_names = list(clients_batched.keys())#randomize client data - using keys
#     client_names = random.sample(all_client_names, k=10)
#     #print(client_names, len(client_names))
#     random.shuffle(client_names)
    
#     average_weights = sum_scaled_weights(scaled_local_weight_list)
#     global_model.set_weights(average_weights) #update global model 

#     for(X_test, Y_test) in test_batched:
#         global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
#         global_acc_list.append(global_acc)
#         global_loss_list.append(global_loss)