# ejemplo de regresión logística

# librerías 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.special import expit

# generacion de datos de forma aleatoria
xmin,xmax=-5,5
n_samples=100
np.random.seed(0)
X=np.random.normal(size=n_samples)
#Y=(X>0).astype(np.float)
Y = (X > 0).astype(float)

X[X>0]*=4

X+=.3*np.random.normal(size=n_samples)

X=X[:,np.newaxis]


# se define el modelo de regresión logística
regresor=linear_model.LogisticRegression(C=1e5)

# entrenamos el modelo
regresor.fit(X,Y)

# visulalización de datos
# Visualizamos el resultado.
plt.figure(1, figsize=(4, 3))
plt.clf()
plt.scatter(X.ravel(), Y, color='black', zorder=20)
X_test = np.linspace(-5, 10, 300)

loss = expit(X_test * regresor.coef_ + regresor.intercept_).ravel()
plt.plot(X_test, loss, color='red', linewidth=3)

# guardado de la imagen y presentacion
plt.savefig("logistic_plot.png", dpi=150, bbox_inches="tight")
plt.show(block=True)  

