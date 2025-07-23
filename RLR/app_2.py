#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importar todas las librerías requeridas:
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn import linear_model, datasets

# Generamos datos aleatorios con una sola variable
n_samples = 1000
X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

# Añadimos al conjunto aleatorio de datos un conjunto de datos Outliers
n_outliers = 50
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# Definimos el modelo lineal Robusto (RLM) con algortimo Huber
regressor_huber = sm.RLM(y, X, M = sm.robust.norms.HuberT())
# Se ajusta al modelo utilizando mínimos cuadrados repesados ​​de forma iterativa.
hub_results = regressor_huber.fit()

# Definimos el modelo lineal 
regressor_Lineal = linear_model.LinearRegression()
# Se ajusta al modelo utilizando mínimos cuadrados
regressor_Lineal.fit(X, y)

#  Definimos el modelo lineal Robusto (RLM) con algoritmo RANSAC
regressor_RANSAC = linear_model.RANSACRegressor()
regressor_RANSAC.fit(X, y)
inlier_mask = regressor_RANSAC.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
y_pred_Lineal = regressor_Lineal.predict(X)
y_pred_RANSAC = regressor_RANSAC.predict(X)
y_pred_Huber = hub_results.fittedvalues

# Comparamos los coeficientes estimados por cada uno de los algoritmos
data = [coef, regressor_Lineal.coef_[0], regressor_RANSAC.estimator_.coef_[0], hub_results.params[0]]
index = ['real', 'Regresion lineal', 'RANSAC','Huber']
coeff = pd.DataFrame(data=data, index=index) 

#  Visualizamos
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(X, y_pred_Lineal, color='navy', linewidth=2, label='Linear regressor')
plt.plot(X, y_pred_RANSAC, color='cornflowerblue', linewidth=2,
         label='RANSAC regressor')
plt.plot(X, y_pred_Huber, color='red', linewidth=2,
          label='Huber regressor')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()