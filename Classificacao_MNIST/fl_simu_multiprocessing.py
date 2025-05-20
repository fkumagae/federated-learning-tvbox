# -----------------------------------------------------------------------------
# Simulação de aprendizado federado (Federated Learning) usando Flower (flwr)
# e o dataset MNIST. Este script executa o servidor e múltiplos clientes em
# processos separados (multiprocessing), com logging detalhado e controle de
# verbosidade. Útil para simulações mais realistas e robustas.
# -----------------------------------------------------------------------------

import multiprocessing
import flwr as fl
import tensorflow as tf
import numpy as np
import time
import logging
import os

# 🔹 Configuração dos Logs (Controle de Verbosidade)
logging.basicConfig(level=logging.INFO)  # Altere para DEBUG, INFO, WARNING ou ERROR

# Reduz logs do TensorFlow (para não poluir o terminal)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0: Tudo, 1: INFO, 2: WARNING, 3: ERROR
tf.get_logger().setLevel("ERROR")

# Configurações globais
num_clients = 3  # Número de clientes simulados
global_epochs = 2  # Épocas de treinamento por cliente

# 🔹 Passo 1: Carregar o dataset (MNIST)
def get_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalização
    return (x_train, y_train), (x_test, y_test)

# 🔹 Passo 2: Criar um modelo simples
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 🔹 Passo 3: Definir a classe do Cliente Federado
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data

    def get_parameters(self, config=None):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        logging.info(f"Treinando Cliente com {len(self.train_data[0])} amostras.")
        self.model.fit(self.train_data[0], self.train_data[1], epochs=global_epochs, batch_size=32, verbose=1)  # verbose=1 para mostrar progresso
        return self.model.get_weights(), len(self.train_data[0]), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        logging.info(f"Avaliação do Cliente -> Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return loss, len(self.test_data[0]), {"accuracy": accuracy}

# 🔹 Passo 4: Criar múltiplos clientes simulados
(x_train, y_train), (x_test, y_test) = get_data()
clients = [
    FLClient(create_model(), (x_train[i::num_clients], y_train[i::num_clients]), (x_test, y_test))
    for i in range(num_clients)
]

# 🔹 Passo 5: Criar função para iniciar o servidor FL
def start_server():
    logging.info("🟢 Iniciando Servidor FL...\n")
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))

# 🔹 Passo 6: Criar função para iniciar um cliente FL
def start_client(client_id):
    logging.info(f"🔵 Iniciando Cliente {client_id}...\n")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=clients[client_id].to_client()
    )

if __name__ == "__main__":
    # Criar processo para o servidor
    server_process = multiprocessing.Process(target=start_server)

    # Criar processos para os clientes
    client_processes = [multiprocessing.Process(target=start_client, args=(i,)) for i in range(num_clients)]

    # Iniciar o servidor
    server_process.start()

    # Aguardar um pouco para garantir que o servidor esteja pronto
    time.sleep(2)

    # Iniciar os clientes
    for p in client_processes:
        p.start()

    # Esperar os clientes terminarem
    for p in client_processes:
        p.join()

    # Finalizar o servidor
    server_process.join()
