import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Carregar os dados
beers = pd.read_csv('beers.csv')

# Remover colunas irrelevantes
beers.drop(['brewery_id', 'name', 'id'], axis=1, inplace=True)

# Remover valores nulos
beers.dropna(inplace=True)

# Separar rótulos (y) e características (x)
y = beers['style']
x = beers.drop('style', axis=1)

# Remover ou agrupar classes raras
rare_classes = y.value_counts()[y.value_counts() < 3].index
beers = beers[~beers['style'].isin(rare_classes)]
y = beers['style']
x = beers.drop('style', axis=1)

# Normalizar as características
x = StandardScaler().fit_transform(x)

# Treinar o modelo Naive Bayes
modelo = GaussianNB()
skfold = StratifiedKFold(n_splits=2)  # Ajustar para evitar classes pequenas
resultado = cross_val_score(modelo, x, y, cv=skfold)

# Resultado
print(f'Acurácia média (validação cruzada): {resultado.mean()}')
