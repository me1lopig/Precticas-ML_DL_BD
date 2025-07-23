
# ejemplo de regresion simple

# importacion de librerias

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import statsmodels.api as sm



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model,datasets


# leemos los datos del archivo
dataset=pd.read_csv('data.csv')

# observamos los datos
muestras,features=dataset.shape

# Obtenemos la información estadistica de cada feature
resumen_estadistico=dataset.describe().T.round(2)
print(resumen_estadistico.iloc[0:13]) 

# guardado del resumen estadistico (opcional)
#filename = 'resumen_estadistico.csv'
#resumen_estadistico.to_csv(filename, index=True)

# regresion lineal simple

# Toma de las variables de entrada X y salida Y
# Toma de las variables de entrada X y salida Y
X = dataset[['MinTemp']] 
# scikit-learn exige que la matriz de entrada X sea bidimensional (n filas × n columnas), aun cuando solo tengas una característica:
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


# Definimos el modelo de predicción HUBER
regressor_HUBER=sm.RLM(Y,X,M=sm.robust.norms.HuberT())
# ajuste de mínimos cuadrados
hub_results=regressor_HUBER.fit()


# Creamos el modelo de regresion lineal
regressor_LINEAL=LinearRegression()
regressor_LINEAL.fit(X_train,Y_train) # con el conjunto de datos de entrenamiento

# modelos de predicción RAMSAC
regressor_RANSAC=linear_model.RANSACRegressor()
regressor_RANSAC.fit(X,Y)
inlier_mask=regressor_RANSAC.inlier_mask_
outlier_mask=np.logical_not(inlier_mask)


# prediccion con los distintos modelos
Y_pred_Lineal=regressor_LINEAL.predict(X)
Y_pred_Huber=hub_results.fittedvalues
Y_pred_Ransac=regressor_RANSAC.predict(X)


#  Visualizamos
plt.scatter(X[inlier_mask], Y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], Y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(X, Y_pred_Lineal, color='navy', linewidth=2, label='Linear regressor')
plt.plot(X, Y_pred_Ransac, color='cornflowerblue', linewidth=2,
         label='RANSAC regressor')
plt.plot(X, Y_pred_Huber, color='red', linewidth=2,
          label='Huber regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()