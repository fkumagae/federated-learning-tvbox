# -----------------------------------------------------------------------------
# Script para simulação de aprendizado federado usando um Autoencoder denso
# para detecção de anomalias em dados pluviométricos. Utiliza Flower para
# orquestrar múltiplos clientes e avalia o desempenho do modelo em ambiente
# federado.
# -----------------------------------------------------------------------------

import multiprocessing
import flwr as fl
import tensorflow as tf
import numpy as np
import time
import logging
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Configurar logs
logging.basicConfig(level=logging.DEBUG)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.get_logger().setLevel("INFO")

# Parâmetros globais
num_clients = 3
global_epochs = 5

# Carregar dados normalizados do CSV (suponha que esteja no mesmo diretório)
df = pd.read_csv("../data/Dataset_Anomalia.csv")
df.columns = df.columns.str.strip()

# Selecionar features e interpolar
features = [
    "Precipitacao_Total",
    "Pressao_Atmosferica",
    "Radiacao_Global",
    "Temperatura_Ar",
    "Umidade_Relativa"
]
df[features] = df[features].interpolate(method='linear', limit_direction='both')

# Normalizar
scaler = StandardScaler()
normalized = scaler.fit_transform(df[features])
for i, col in enumerate(features):
    df[col + "_norm"] = normalized[:, i]

# Separar dados normais
df["Anomalia_Pluviometrica"] = df["Anomalia_Pluviometrica"].astype(str)
df_normal = df[df["Anomalia_Pluviometrica"] == "0"]

# Usar apenas dados normalizados
selected_cols = [col + "_norm" for col in features]
X = df_normal[selected_cols].values.astype(np.float32)
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Dividir dados entre clientes
client_data = [
    (X_train[i::num_clients], X_test)
    for i in range(num_clients)
]

# Modelo Autoencoder
def create_autoencoder(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Classe do cliente FLWR
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config=None):
        logging.debug("Obtendo parâmetros do modelo.")
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        logging.info("====================")
        logging.info(f"[FIT] Iniciando treinamento com {len(self.train_data)} amostras...")
        start_time = time.time()
        history = self.model.fit(
            self.train_data,
            self.train_data,
            epochs=global_epochs,
            batch_size=32,
            verbose=2
        )
        end_time = time.time()
        logging.info(f"[FIT] Treinamento finalizado em {end_time - start_time:.2f} segundos.")
        return self.model.get_weights(), len(self.train_data), {}
        
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        logging.info("====================")
        logging.info("[EVAL] Iniciando avaliação no conjunto de teste...")
        start_time = time.time()
        loss = self.model.evaluate(self.test_data, self.test_data, verbose=2)
        end_time = time.time()
        logging.info(f"[EVAL] Avaliação finalizada em {end_time - start_time:.2f} segundos. Loss (Reconstr.): {loss:.4f}")
        return loss, len(self.test_data), {"mse": loss}

# Inicializa clientes
input_dim = X.shape[1]
clients = [
    FLClient(create_autoencoder(input_dim), train, test)
    for train, test in client_data
]

def start_server():
    logging.info("\n======================\n🟢 Servidor Federado Iniciado\n======================")
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5))

def start_client(client_id):
    logging.info(f"\n🔵 Cliente {client_id} Iniciado")
    fl.client.start_client(server_address="127.0.0.1:8080", client=clients[client_id].to_client())

if __name__ == "__main__":
    server_process = multiprocessing.Process(target=start_server)
    client_processes = [
        multiprocessing.Process(target=start_client, args=(i,)) for i in range(num_clients)
    ]

    server_process.start()
    time.sleep(2)
    for p in client_processes:
        p.start()
    for p in client_processes:
        p.join()
    server_process.join()
