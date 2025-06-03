import logging
import flwr as fl
from utils import ROUNDS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    logging.info(f"🟢 Iniciando servidor FL com {ROUNDS} rounds...")
    strategy = fl.server.strategy.FedAvg()
    # Inicia o servidor (sem salvar modelo global automaticamente)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )
    logging.info("Servidor finalizado. Se desejar salvar o modelo global, implemente a lógica em um dos clientes ou utilize uma versão mais recente do Flower.")

if __name__ == "__main__":
    main()
