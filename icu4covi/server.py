import flwr as fl

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3), server_address="172.16.1.21:8080")