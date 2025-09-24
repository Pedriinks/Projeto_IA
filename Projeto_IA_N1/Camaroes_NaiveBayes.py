import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Fazer a leitura do arquivo .csv
df = pd.read_csv("Estoque_Camaroes.csv", delimiter = ";")

# Definindo a coluna alvo
target_col = 'Apresentação do produto'
X = df.drop(columns = [target_col])
y = df[target_col]

# Transformar colunas categóricas em númericas
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

# Codifica a variável alvo
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Criar e treinar o modelo Naive Bayes
model = GaussianNB()
model.fit(X, y)

# Fazer uma previsão com os mesmos dados
prediction = model.predict(X)
predicted_label = le_target.inverse_transform(prediction)

print("\nPredição sobre a apresentação do produto:", predicted_label[0])

# Analise da matriz de confusao
cm = confusion_matrix(y, prediction)

print("A matriz de confusão é:\n", cm)

# Acurácia
accuracy = accuracy_score(y, prediction)
print(f"Acurácia do modelo: {accuracy:.2%}")