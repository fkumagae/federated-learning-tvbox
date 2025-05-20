import logging
import flwr as fl
from utils import ROUNDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info(f"🟢 Iniciando servidor FL com {ROUNDS} rounds...")
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
