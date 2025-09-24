import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree

# Faz a leitura do arquivo .csv
df = pd.read_csv("Estoque_Camaroes.csv", delimiter = ";")

# Remove a coluna Data de envio
df = df.drop(columns = ["Data de envio"])

# Converte as colunas de tipo categoricas para tipo numericas
label_encoders = {}
for column in df.select_dtypes(include = ["object"]).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Transformando o tipo dessas colunas para numérico e trocando "," por "."
df["Camarão Rosa (Kg)"] = df["Camarão Rosa (Kg)"].replace(",", ".", regex = True)
df["Camarão Rosa (Kg)"] = df["Camarão Rosa (Kg)"].apply(pd.to_numeric)

df["Camarão Sete Barbas (Kg)"] = df["Camarão Sete Barbas (Kg)"].replace(",", ".", regex = True)
df["Camarão Sete Barbas (Kg)"] = df["Camarão Sete Barbas (Kg)"].apply(pd.to_numeric)

df["Camarão Branco (Kg)"] = df["Camarão Branco (Kg)"].replace(",", ".", regex = True)
df["Camarão Branco (Kg)"] = df["Camarão Branco (Kg)"].apply(pd.to_numeric)

df["Camarão Santana ou Vermelho (Kg)"] = df["Camarão Santana ou Vermelho (Kg)"].replace(",", ".", regex = True)
df["Camarão Santana ou Vermelho (Kg)"] = df["Camarão Santana ou Vermelho (Kg)"].apply(pd.to_numeric)

df["Camarão Barba-ruça (Kg)"] = df["Camarão Barba-ruça (Kg)"].replace(",", ".", regex = True)
df["Camarão Barba-ruça (Kg)"] = df["Camarão Barba-ruça (Kg)"].apply(pd.to_numeric)

df = df.select_dtypes(include=['number'])

# Usando a coluna de Apresentação do produto para a previsao
X = df.drop("Apresentação do produto", axis = 1)
y = df["Apresentação do produto"]

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1) #random_state = 1 fixa a semente do gerador aleatório

# Criar o classificador de arvore de decisao
decision_T = DecisionTreeClassifier(max_leaf_nodes = 4, random_state = 1)

# Treinar o modelo
decision_T = decision_T.fit(X_train, y_train)

# Plotar a árvore de decisão
plt.figure(figsize=(12, 8))
plot_tree(decision_T, filled = True, class_names = [str(cls) for cls in df["Apresentação do produto"].unique()])
plt.show()

# Prever a resposta para os dados de teste
y_pred = decision_T.predict(X_test)
print("\nAcurácia:", metrics.accuracy_score(y_test, y_pred))