import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import flwr as fl
import logging

# Parâmetros do FL
WINDOW_SIZE = 24
NUM_CLIENTS = 3
EPOCHS = 5
BATCH_SIZE = 32
ROUNDS = 3

# Carregamento e pré-processamento de dados
def load_data(csv_path):
    logging.info(f"🔄 Carregando dados de: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    data = df.values
    return data

# Cria sequências para Autoencoder LSTM
def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

# Cria o modelo LSTM Autoencoder
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

# Cliente FL para Autoencoder LSTM
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data):
        self.model = model
        self.train_data = train_data

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        logging.info(f"📦 Treinando com {len(self.train_data)} sequências por {EPOCHS} épocas...")
        self.model.fit(
            self.train_data, self.train_data,
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0
        )
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.train_data, self.train_data, verbose=0)
        logging.info(f"Avaliação — MSE: {loss:.4f}")
        return loss, len(self.train_data), {"mse": loss}

# Prepara dados para um cliente específico
def prepare_client_data(csv_path, client_id):
    data = load_data(csv_path)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    sequences = create_sequences(scaled, WINDOW_SIZE)
    part = sequences[client_id::NUM_CLIENTS]
    return part, sequences.shape[2]
