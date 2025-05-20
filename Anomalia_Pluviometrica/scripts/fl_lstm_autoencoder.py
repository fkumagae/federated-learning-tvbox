# -----------------------------------------------------------------------------
# Script para simulação de aprendizado federado com Autoencoder LSTM para
# detecção de anomalias em séries temporais pluviométricas. Utiliza Flower
# para orquestrar múltiplos clientes e salva o modelo treinado ao final.
# -----------------------------------------------------------------------------

import multiprocessing
import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import logging
import time

# 🔧 Verbosidade (DEBUG, INFO, WARNING, ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 🔧 Parâmetros
WINDOW_SIZE = 24
NUM_CLIENTS = 3
EPOCHS = 5
BATCH_SIZE = 32
ROUNDS = 3

# 🔹 Funções utilitárias
def load_data(csv_path):
    logging.info(f"🔄 Carregando dados de: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    logging.info(f"Dados carregados com shape: {df.shape}")
    return df.values

def create_sequences(data, window_size):
    logging.info(f"Criando janelas de tamanho {window_size}")
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    logging.info(f"Total de janelas criadas: {len(sequences)}")
    return np.array(sequences)

def create_model(n_features):
    logging.info(f"🧠 Criando modelo LSTM Autoencoder com {n_features} features")
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE, n_features)),
        tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=False),
        tf.keras.layers.RepeatVector(WINDOW_SIZE),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=True),
        tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# 🔹 Classe Cliente FL
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data):
        self.model = model
        self.train_data = train_data

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        logging.info(f"📦 Cliente treinando {len(self.train_data)} sequências por {EPOCHS} épocas...")
        self.model.fit(self.train_data, self.train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.train_data, self.train_data, verbose=0)
        logging.info(f"Avaliação do cliente — MSE: {loss:.4f}")
        return loss, len(self.train_data), {"mse": loss}

# 🔹 Carregar dados e preparar partições
data = load_data("../data/Dataset_Anomalia.csv")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
sequences = create_sequences(scaled_data, WINDOW_SIZE)

clients = []
for i in range(NUM_CLIENTS):
    part = sequences[i::NUM_CLIENTS]
    model = create_model(n_features=sequences.shape[2])
    client = FLClient(model, part)
    logging.info(f"👤 Cliente {i} criado com {part.shape[0]} janelas.")
    clients.append(client)

# 🔹 Funções para servidor e clientes
def start_server():
    logging.info(f"🟢 Servidor iniciado com {ROUNDS} rounds de FL...")
    strategy = fl.server.strategy.FedAvg()
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
        strategy=strategy
    )

def start_client(client_id):
    logging.info(f"🔵 Iniciando cliente {client_id}...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=clients[client_id].to_client()
    )

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    server = multiprocessing.Process(target=start_server)
    client_procs = [
        multiprocessing.Process(target=start_client, args=(i,))
        for i in range(NUM_CLIENTS)
    ]

    server.start()
    time.sleep(2)  # Esperar o servidor iniciar

    for p in client_procs:
        p.start()
    for p in client_procs:
        p.join()

    server.join()
# 💾 Salvar o modelo do cliente 0 após o FL
logging.info("💾 Salvando o modelo do cliente 0 como 'fl_lstmAE.keras'")
clients[0].model.save("/Users/felipekumagae/LINCE/Projetos/Federated_Learning/Anomalia_Pluviometrica/models/fl_lstmAE.keras")