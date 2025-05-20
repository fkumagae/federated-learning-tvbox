# -----------------------------------------------------------------------------
# Template de projeto para detecção de anomalias pluviométricas
# usando Federated Learning com LSTM Autoencoder.
# -----------------------------------------------------------------------------

Este projeto implementa um pipeline FL para detectar anomalias em séries temporais pluviométricas.

Estrutura:
- data/: datasets (ex: Dataset_Anomalia.csv)
- models/: modelos treinados pós-treinamento federado
- scripts/: scripts de servidor, cliente e utilitários
- requirements.txt: dependências Python
- README.md: documentação deste projeto

Deploy em TVBOXs:
1. Copie toda a pasta `Anomalia_Template/` para cada TVBOX.
2. Em uma TVBOX (servidor):
   - Instale dependências: `pip install -r requirements.txt`
   - Rode o servidor FL: `python scripts/run_fl_server.py`
3. Nas demais TVBOXs (clientes):
   - Instale dependências: `pip install -r requirements.txt`
   - Identifique um ID único para cada cliente (0 a NUM_CLIENTS-1).
   - Rode o cliente: `python scripts/run_fl_client.py --client_id <ID>`
4. O servidor orquestra rounds de federated learning entre clientes conectados.
5. Ao final, o modelo global treinado estará disponível na pasta `models/` do servidor.
