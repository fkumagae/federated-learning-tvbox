# -----------------------------------------------------------------------------
# Script para análise de desempenho de um Autoencoder LSTM treinado para detecção
# de anomalias em séries temporais pluviométricas. Gera gráficos de erro de
# reconstrução, matriz de confusão, curva precision-recall e salva relatório de
# classificação, utilizando um modelo treinado e dados reais.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score
import seaborn as sns

# Carregamento dos dados processados
df = pd.read_csv("../data/Dataset_Anomalia.csv")
df = df.select_dtypes(include=[np.number])
df.fillna(df.mean(), inplace=True)

from sklearn.preprocessing import StandardScaler
WINDOW_SIZE = 24
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.values)

# Recriar janelas
def create_sequences(data, window_size):
    return np.array([data[i:i+window_size] for i in range(len(data) - window_size + 1)])

sequences = create_sequences(scaled_data, WINDOW_SIZE)

# Carregar modelo treinado (exemplo: fl_lstm_autoencoder localmente treinado)
from tensorflow.keras.models import load_model
model = load_model("/Users/felipekumagae/LINCE/Projetos/Federated_Learning/models/fl_lstmAE.h5") #escolher Modelo LSTM AE

# Fazer reconstrução
X_pred = model.predict(sequences)
mse = np.mean(np.power(sequences - X_pred, 2), axis=(1,2))

# Criar threshold usando percentil
threshold = np.percentile(mse, 95)

# Geração dos rótulos reais ajustados
y_true = df['Anomalia_Pluviometrica'].values[WINDOW_SIZE - 1:]
y_pred = (mse > threshold).astype(int)

# === 1. MSE vs Threshold ===
plt.figure(figsize=(12, 4))
plt.plot(mse, label="Erro de reconstrução (MSE)")
plt.axhline(y=threshold, color='r', linestyle='--', label=f"Threshold (95º percentil) = {threshold:.4f}")
plt.title("Erro de reconstrução ao longo do tempo")
plt.xlabel("Índice da janela")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_mse_threshold.png")
plt.close()

# === 2. Original vs Reconstruído (uma janela específica)
sample_idx = 100  # índice da janela a visualizar
feature_idx = 0   # índice da feature a ser exibida

plt.figure(figsize=(10, 4))
plt.plot(sequences[sample_idx,:,feature_idx], label="Original")
plt.plot(X_pred[sample_idx,:,feature_idx], label="Reconstruído")
plt.title(f"Comparação: Original vs Reconstruído (feature {feature_idx})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_reconstruction_example.png")
plt.close()

# === 3. Matriz de confusão + classification report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomalia"], yticklabels=["Normal", "Anomalia"])
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.savefig("plot_confusion_matrix.png")
plt.close()

# Salvar relatório em texto
with open("classification_report.txt", "w") as f:
    f.write(report)

# === 4. Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_true, mse)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_precision_recall_curve.png")
plt.close()
