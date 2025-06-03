# -----------------------------------------------------------------------------
# Projeto de detecção de anomalias pluviométricas
# Federated Learning com LSTM Autoencoder
# -----------------------------------------------------------------------------

Este projeto implementa um pipeline de Federated Learning (FL) para detectar anomalias em séries temporais pluviométricas.

## Estrutura do Projeto
- `data/`: datasets (ex: `Dataset_Anomalia.csv`)
- `models/`: modelos treinados após o treinamento federado
- `scripts/`: scripts de servidor, cliente e utilitários
- `requirements.txt`: dependências Python
- `README.md`: documentação deste projeto

## Passo a passo para execução

### 1. Pré-requisitos
- **Python 3.10** instalado (TensorFlow não suporta Python 3.13+)
- Recomenda-se usar ambiente virtual para isolar as dependências

### 2. Crie e ative o ambiente virtual
No terminal, dentro da pasta `Anomalia_TVBOX`:

```sh
python3.10 -m venv fl_venv
source fl_venv/bin/activate
```

### 3. Instale as dependências
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Execute o servidor e os clientes

No terminal, sempre a partir da raiz `Anomalia_TVBOX`:

```sh
cd /caminho/para/Anomalia_TVBOX
source fl_venv/bin/activate
python scripts/run_fl_server.py
# Em outro terminal, para cada cliente:
python scripts/run_fl_client.py --client_id <ID>
```


### 4. Resultados
- O servidor orquestra os rounds de federated learning entre os clientes conectados.
- Ao final, o modelo global treinado estará disponível na pasta `models/` do servidor (se implementado).

---
**Dicas:**
- Sempre execute os scripts a partir da pasta `Anomalia_TVBOX` para evitar erros de caminho.
- Se aparecer erro de espaço em disco, libere espaço antes de instalar dependências.
- Para verificar a versão do Python ativo: `python --version`
- Para verificar a versão do TensorFlow: `python -c "import tensorflow as tf; print(tf.__version__)"`
