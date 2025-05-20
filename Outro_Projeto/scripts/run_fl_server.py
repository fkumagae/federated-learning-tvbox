import flwr as fl
import logging

def start_server():
    logging.basicConfig(level=logging.INFO)
    logging.info("Iniciando servidor federado...")
    fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=3))

if __name__ == "__main__":
    start_server()
