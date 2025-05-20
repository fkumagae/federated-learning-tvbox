import argparse
import logging
import flwr as fl
from utils import prepare_client_data, WINDOW_SIZE, EPOCHS, BATCH_SIZE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main(client_id):
    csv_path = "../data/Dataset_Anomalia.csv"
    train_data, n_features = prepare_client_data(csv_path, client_id)
    model = create_model(n_features)
    client = FLClient(model, train_data)

    logging.info(f"🔵 Iniciando cliente {client_id}...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    args = parser.parse_args()
    main(args.client_id)
