# 🌼 Federated Learning Simulation with Flower + TensorFlow

**Simulação local de Aprendizado Federado com Flower, TensorFlow (exemplo com MNIST).**

Este repositório reúne múltiplos **subprojetos de Aprendizado Federado (Federated Learning)**, cada um com um objetivo distinto (ex: detecção de anomalias, classificação com MNIST etc.).  
Todos os projetos utilizam o framework [Flower (FLWR)](https://flower.dev/) com TensorFlow, simulando múltiplos clientes locais.

[![Python](https://img.shields.io/badge/python-3.8--3.10-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange)](https://www.tensorflow.org/)
[![Flower](https://img.shields.io/badge/Flower-1.17.0-brightgreen)](https://flower.dev)

---

## 📁 Organização dos subprojetos

Cada subpasta corresponde a uma aplicação distinta de Federated Learning:

- `Anomalia_Pluviometrica/`: Detecção de anomalias em séries temporais ambientais
- `Classificacao_MNIST/`: Classificação de dígitos com o dataset MNIST

Cada subprojeto possui:

- `scripts/`: scripts de simulação federada
- `data/`: datasets utilizados
- `requirements.txt`: dependências do subprojeto
- `README.md`: instruções específicas

---

## 🧰 Requisitos

- Python **3.8** a **3.11**
- `pip` atualizado
- Ambiente virtual recomendado (`venv`)

---

## 🖥️ Instalação passo a passo (Windows / macOS)

> 💡 Recomendado: usar **Python 3.10** para garantir compatibilidade com TensorFlow
sim
| Etapa | macOS (Terminal) | Windows (CMD / PowerShell) |
|-------|------------------|-----------------------------|
| 1. Clone o repositório | `git clone https://github.com/felipekumagae/federated-learning-tvbox.git`<br>`cd federated-learning-tvbox` | idem |
| 2. Crie o ambiente virtual | `python3.10 -m venv fl_env` | `python -m venv fl_env` |
| 3. Ative o ambiente | `source fl_env/bin/activate` | `fl_env\Scripts\activate` ou `.\fl_env\Scripts\Activate.ps1` |
| 4. Atualize o pip | `pip install --upgrade pip` | idem |
| 5. Instale as dependências | `pip install -r requirements.txt` | idem |
| 6. Navegue até o subprojeto e rode a simulação Exemplo (MNIST)| `cd Classificacao_MNIST/scripts`<br>`python fl_simu.py` | idem |
| 7. Finalize (opcional) | `deactivate` | idem |

---

## 🚀 O que a simulação faz 

- Inicia um **servidor federado local** (`localhost:8080`)
- Executa múltiplos clientes com diferentes subconjuntos de dados
- Cada cliente **treina localmente** e envia os pesos ao servidor
- O servidor realiza a **agregação federada via média**

---

## 📁 Estrutura do Projeto

```bash
Federated_Learning/ ├── Anomalia_Pluviometrica/ │ ├── scripts/ │ ├── data/ │ ├── requirements.txt │ └── README.md ├── Classificacao_MNIST/ │ ├── scripts/ │ ├── data/ │ ├── requirements.txt │ └── README.md ├── fl_env/ # Ambiente virtual (ignorado pelo Git) ├── requirements.txt # Dependências globais (opcional) └── README.md # Este arquivo
```

---

## ⚙️ Personalizações possíveis

Nos scripts, altere:

```python
num_clients = 3         # Número de clientes
num_rounds = 5          # Rounds globais
local_epochs = 1        # Épocas locais por cliente
```

---

## 🧪 Testado com

| Componente   | Versão         |
|--------------|----------------|
| Python       | 3.10 ✅       |
| TensorFlow   | 2.19.0 ✅       |
| Flower       | 1.17.0 ✅        |
| macOS        | Monterey 12+ ✅ |
| Windows      | 10/11 ✅        |

---

## ⚠️ Observações

- Python 3.13 ainda **não é compatível** com TensorFlow.
- Verifique se o `pip` está atualizado antes de instalar as libs.

✅ Regras práticas para definir ROUNDS e EPOCHS

## Situação	Estratégia recomendada
- Poucos dados por cliente	Aumentar EPOCHS, reduzir ROUNDS
- Muitos dados por cliente	Reduzir EPOCHS, aumentar ROUNDS
- Conexão instável ou custo de comunicação alto	Treinar mais localmente (EPOCHS ↑)
- Datasets homogêneos entre clientes	EPOCHS=1~3 e ROUNDS=20+ funcionam bem
- Datasets heterogêneos (non-IID)	EPOCHS=1 e ROUNDS mais altos (50+)
---

## 👥 Autoria

Desenvolvido por **Felipe Kumagae - LINCE (Liga de Inteligência Neuro-Computacional na Engenharia)**  
📍 Instituto de Ciência e Tecnologia de Sorocaba – UNESP  
🔗 https://github.com/felipekumagae/federated-learning-tvbox

---

## 📦 Deploy em TVBOXs

1. Copie o subprojeto desejado (ex: `Anomalia_Pluviometrica/`) para cada TV BOX.
2. **TV BOX Servidor**:
   - Instale dependências: `pip install -r requirements.txt`
   - Inicie o servidor federado: `python scripts/run_fl_server.py`
3. **TV BOXs Clientes**:
   - Instale dependências: `pip install -r requirements.txt`
   - Para cada cliente, use um `client_id` único (0..NUM_CLIENTS-1)
   - Inicie o cliente: `python scripts/run_fl_client.py --client_id <ID>`
4. O servidor orquestra os rounds e, ao final, salva o modelo global em `models/` na pasta do subprojeto.
