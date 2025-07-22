
# ejemplo de regresion multiple

# importacion de librerias

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as seabornInstance


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# leemos los datos del archivo
dataset=pd.read_csv('winequality-red.csv')

# limpiamos el dataset de valores NaN
dataset.dropna(inplace=True)


# informacion estadistica de las variables
info_estadistica=dataset.describe()
print(info_estadistica)


# Capture column names
variables = dataset.columns.tolist()
#print('Variables en el dataset:')
#print(variables)

# asignacion de variables
X = dataset.drop('quality', axis=1)   # todas las columnas menos 'quality'
Y = dataset['quality']                # la columna objetivo

# dividimos los conjuntos de datos
# dividimos los datos en un 80% de entrenamiento y un 20% de comprobación
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Creamos el modelo de regresion lineal

regressor=LinearRegression()
regressor.fit(X_train,Y_train) # con el conjunto de datos de entrenamiento

# comprobacion de los coeficienets con cada una de las variables
coef_lineales = pd.DataFrame({'Coeficientes más óptimos': regressor.coef_}, index=X.columns)
print(coef_lineales)

# predicción sobre el conjunto de prueba
Y_pred=regressor.predict(X_test)

# diferencia entre el valor predicho y el real
df=pd.DataFrame({'Dato':Y_test,'Predicción':Y_pred})
print(df.head(50)) # primeros valores



