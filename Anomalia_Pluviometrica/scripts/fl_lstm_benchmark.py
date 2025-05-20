# -----------------------------------------------------------------------------
# Script para benchmark de aprendizado federado usando LSTM em séries temporais.
# Realiza experimentos variando épocas e rodadas, avaliando o desempenho médio
# (MSE) entre múltiplos clientes simulados com Flower e TensorFlow.
# -----------------------------------------------------------------------------

import multiprocessing
import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# 🔧 Parâmetros fixos
WINDOW_SIZE = 24
NUM_CLIENTS = 3
BATCH_SIZE = 32
COMBINATIONS = [(1, 50), (3, 20), (5, 15), (10, 10)]

# 🔹 Funções utilitárias
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=[np.number])
    df.fillna(df.mean(), inplace=True)
    return df.values

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size + 1):
        sequences.append(data[i:i + window_size])
    return np.array(sequences)

def create_model(n_features):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation="relu", input_shape=(WINDOW_SIZE, n_features), return_sequences=True),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=False),
        tf.keras.layers.RepeatVector(WINDOW_SIZE),
        tf.keras.layers.LSTM(32, activation="relu", return_sequences=True),
        tf.keras.layers.LSTM(64, activation="relu", return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, epochs):
        self.model = model
        self.train_data = train_data
        self.epochs = epochs

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_data, self.train_data,
                       epochs=self.epochs, batch_size=BATCH_SIZE, verbose=0)
        return self.model.get_weights(), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss = self.model.evaluate(self.train_data, self.train_data, verbose=0)
        return loss, len(self.train_data), {"mse": loss}

# 🔁 Benchmark
if __name__ == "__main__":
    csv_path = "../data/Dataset_Anomalia.csv"
    data = load_data(csv_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    sequences = create_sequences(scaled_data, WINDOW_SIZE)

    results = []

    for EPOCHS, ROUNDS in COMBINATIONS:
        print(f"Rodando FL com EPOCHS={EPOCHS} e ROUNDS={ROUNDS}...")

        clients = []
        for i in range(NUM_CLIENTS):
            part = sequences[i::NUM_CLIENTS]
            model = create_model(n_features=sequences.shape[2])
            client = FLClient(model, part, epochs=EPOCHS)
            clients.append(client)

        def start_server():
            strategy = fl.server.strategy.FedAvg(
                min_fit_clients=NUM_CLIENTS,
                min_eval_clients=NUM_CLIENTS,
                min_available_clients=NUM_CLIENTS,
            )
            fl.server.start_server(server_address="127.0.0.1:8080", config={"num_rounds": ROUNDS}, strategy=strategy)

        def start_client(client_id):
            fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=clients[client_id])

        multiprocessing.set_start_method("spawn", force=True)
        server = multiprocessing.Process(target=start_server)
        client_procs = [multiprocessing.Process(target=start_client, args=(i,)) for i in range(NUM_CLIENTS)]

        server.start()
        import time; time.sleep(2)

        for p in client_procs:
            p.start()
        for p in client_procs:
            p.join()

        server.join()

        # Avaliação final
        client_mses = [client.model.evaluate(client.train_data, client.train_data, verbose=0) for client in clients]
        avg_mse = np.mean(client_mses)
        results.append((EPOCHS, ROUNDS, avg_mse))
        print(f">>> Média MSE para EPOCHS={EPOCHS}, ROUNDS={ROUNDS}: {avg_mse:.4f}")

    # Resultados finais
    print("\nResumo dos testes:")
    for ep, rd, mse in results:
        print(f"EPOCHS={ep} | ROUNDS={rd} --> MSE médio = {mse:.4f}")
