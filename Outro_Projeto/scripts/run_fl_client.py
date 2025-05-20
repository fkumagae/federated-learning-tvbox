import flwr as fl
import argparse
import logging

# Exemplo de cliente federado simples
class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config=None):
        # Substitua pelo seu modelo real
        return []
    def fit(self, parameters, config):
        return [], 0, {}
    def evaluate(self, parameters, config):
        return 0.0, 0, {}

def main(client_id):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Iniciando cliente federado {client_id}...")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FLClient())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()
    main(args.client_id)
