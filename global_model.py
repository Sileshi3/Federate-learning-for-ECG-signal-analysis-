import tensorflow as tf
import flwr as fl 
strategy=fl.server.strategy.FedAvg()
fl.server.start_server(strategy=strategy,
                       server_address="0.0.0.0:8080",
                       config=fl.server.ServerConfig(num_rounds=3))
