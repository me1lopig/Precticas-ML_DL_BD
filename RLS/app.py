
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
print(resumen_estadistico.iloc[0:13]) 

# guardado del resumen estadistico (opcional)
#filename = 'resumen_estadistico.csv'
#resumen_estadistico.to_csv(filename, index=True)

# regresion lineal simple

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

# dibujo de la recta obtenida por regresión lineal y los datos

plt.scatter(X_test,Y_test,color='gray')
plt.plot(X_test,Y_pred,color='red',linewidth=2)
plt.title('Temperatura mínima vs Temperatura máxima')  
plt.xlabel('Temperatura mínima')  
plt.ylabel('Temperatura máxima')
plt.show()

# cálculo de las métricas

# valores de control
Y_promedio=np.mean(Y)
raiz_ECM=np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)) # raiz del error cuadratico medio


print('Promedio de Y',np.mean(Y))
print('Error Absoluto Medio:',metrics.mean_absolute_error(Y_test, Y_pred)) 
print('Error Cuadratico Medio:', metrics.mean_squared_error(Y_test, Y_pred)) 
print('Raíz del error cuadrático medio:',raiz_ECM )
print('Coeficidiente de correlación:', metrics.r2_score(Y_test,Y_pred))


# control de prediccion
if (raiz_ECM>=0.1*Y_promedio):
    print("El ajuste no se considera bueno, la raiz del error cuadrático medio es mayor el 10 % del la media de variable predicha")
else:
    print('El ajuste se considera bueno')






