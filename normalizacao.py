from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import normalize

X = [[4,1,2,2],[1,3,9,3],[5,7,5,1]]

normalizador = MinMaxScaler(feature_range=(0,1))
print(normalizador.fit_transform(X))

normalizador_StandardScaler = StandardScaler()
print(normalizador_StandardScaler.fit_transform(X))

normalizador_MaxAbsScaler = MaxAbsScaler()
print(normalizador_MaxAbsScaler.fit_transform(X))

normalizador_Normalize = normalize(X, norm='l1')
print(normalizador_Normalize)

normalizador_Normalize = normalize(X, norm='l2')
print(normalizador_Normalize)

normalizador_Normalize = normalize(X, norm='max',axis=0)
print(normalizador_Normalize)
