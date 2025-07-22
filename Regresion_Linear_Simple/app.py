
# ejemplo de regresion simple

# importacion de librerias

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as seabornInstance


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# leemos los datos del archivo
dataset=pd.read_csv('data.csv')

# observamos los datos
muestras,features=dataset.shape

# Obtenemos la información estadistica de cada feature
resumen_estadistico=dataset.describe().T.round(2)
print(resumen_estadistico.iloc[2:4]) 

# guardado del resumen estadistico (opcional)
#filename = 'resumen_estadistico.csv'
#resumen_estadistico.to_csv(filename, index=True)

# regresion lineal simple

# Toma de las variables de entrada X y salida Y
X = dataset[['MinTemp']]
Y = dataset['MaxTemp']

# Visuali<acion de los datos 

dataset.plot(x='MinTemp', y='MaxTemp', style='o')  
plt.title('Temperatura mínima vs Temperatura máxima')  
plt.xlabel('Temperatura mínima')  
plt.ylabel('Temperatura máxima')
plt.legend(loc="lower right", title="Temperatura Máxima") 
plt.show()

# dividimos los datos en un 80% de entranamiento y un 20% de comprobación
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

# Creamos el modelo de regresion lineal

regressor=LinearRegression()

regressor.fit(X_train,Y_train) # con el conjunto de datos de entrenamiento


# obtencion de los paráemtros de la recta de regresión
print('Termino independiente %.3f'%regressor.intercept_)
print('Termino pendiente de la recta %.3f'%regressor.coef_[0])

# predicciones del modelo

Y_pred=regressor.predict(X_test)
Y_test=Y_test.to_numpy()

# Comparacion de los valores 

comparison_table = pd.DataFrame({
    'MinTemp': X_test['MinTemp'].to_numpy(),
    'Actual_MaxTemp': Y_test,
    'Prediccion_MaxTemp': Y_pred,
    'Error absoluto': Y_test - Y_pred
})

print(comparison_table.head(20)) # se imprimen los primeros datos

