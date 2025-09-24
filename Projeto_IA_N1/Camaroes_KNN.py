import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

# Fazer a leitura do arquivo .csv
df = pd.read_csv("Estoque_Camaroes.csv", delimiter = ";")

X = df.drop(columns = ["Apresentação do produto"])
y = df["Apresentação do produto"]

# Aplicando Label Encodin para os dados categóricos
label_encoders = {}
categ_cols = X.select_dtypes(include = ["object"]).columns
for column in categ_cols:
    le = LabelEncoder()
    X[column] = le.fit_transform(X[column].astype(str))
    label_encoders[column] = le

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
print(y_test)

# Treinar o modelo KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)

# Fazer previsões de classe e de probabilidade
y_pred = knn.predict(X_test) # A classe prevista diretamente para cada amostra.
y_pred_proba = knn.predict_proba(X_test) # Para as probabilidades associadas a cada classe para cada amostra

lb = LabelBinarizer()
y_test_binarized = lb.fit_transform(y_test)
print(y_test_binarized)
auc_roc = roc_auc_score(y_test_binarized, y_pred_proba, average='weighted', multi_class='ovr')

# Para Acuracia
accuracy = accuracy_score(y_test, y_pred)

# Para Precisao
precision = precision_score(y_test, y_pred, average = "weighted")

# Para Recall
recall = recall_score(y_test, y_pred, average = "weighted")

# Para F1-Score
f1 = f1_score(y_test, y_pred, average = "weighted")

# Exibindo os resultados
print(f"Acurácia: {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC Score (Área sob a curva ROC): {auc_roc:.4f}")