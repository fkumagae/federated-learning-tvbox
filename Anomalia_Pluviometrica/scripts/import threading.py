# -----------------------------------------------------------------------------
# Script para simulação de aprendizado federado usando o dataset MNIST.
# Executa múltiplos clientes e um servidor federado em threads locais,
# cada cliente treina um modelo de classificação de dígitos em uma partição dos dados.
# Utiliza Flower para orquestração federada.
# -----------------------------------------------------------------------------

import threading
import flwr as fl
import tensorflow as tf
import numpy as np
import time

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

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.train_data[0], self.train_data[1], epochs=global_epochs, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.train_data[0]), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        return loss, len(self.test_data[0]), {"accuracy": accuracy}

# 🔹 Passo 4: Criar múltiplos clientes simulados
(x_train, y_train), (x_test, y_test) = get_data()
clients = [
    FLClient(create_model(), (x_train[i::num_clients], y_train[i::num_clients]), (x_test, y_test))
    for i in range(num_clients)
]

# 🔹 Passo 5: Criar função para iniciar o servidor FL
def start_server():
    print("Iniciando Servidor FL...\n")
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))

# 🔹 Passo 6: Criar função para iniciar um cliente FL
def start_client(client_id):
    print(f"Iniciando Cliente {client_id}...\n")
    fl.client.start_numpy_client("127.0.0.1:8080", client=clients[client_id])

# 🔹 Passo 7: Criar e iniciar threads para servidor e clientes
server_thread = threading.Thread(target=start_server)

client_threads = [threading.Thread(target=start_client, args=(i,)) for i in range(num_clients)]

# Iniciar o servidor em uma thread separada
server_thread.start()

# Esperar o servidor iniciar antes de rodar os clientes
time.sleep(2)

# Iniciar os clientes em threads separadas
for t in client_threads:
    t.start()

# Aguardar a execução de todos os clientes
for t in client_threads:
    t.join()

# Finalizar o servidor
server_thread.join()
