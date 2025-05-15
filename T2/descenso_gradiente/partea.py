import numpy as np

def gradient_descent(X, y, NITMAX, gamma):
    """
    Algoritmo de descenso del gradiente para minimizar:
        J(θ) = (1/2N) * ||Xθ - y||²
    """
    N = X.shape[0]
    theta = np.zeros((2,))  # Inicializar parámetros en cero
    history = []

    for t in range(1, NITMAX + 1):
        # Paso 1: cálculo del gradiente
        y_pred = X @ theta
        error = y_pred - y
        grad = (1/N) * (X.T @ error)

        # Paso 2: actualización con tasa dependiente de t
        theta = theta - gamma(t) * grad

        # Paso 3: guardar el costo cada 10 iteraciones
        if t % 10 == 0:
            cost = (1/(2*N)) * np.linalg.norm(X @ theta - y)**2
            history.append(cost)

    return theta, history


#import pandas as pd
#import matplotlib.pyplot as plt
#
## cargar datos desde CSV
#data = pd.read_csv("datos_lineales.csv")
#X_raw = data[['x']].values
#y = data['y'].values
#
## añadir columna de unos para el término independiente (bias)
#X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])
#
## parámetros del algoritmo
#NITMAX = 100
#
#def gamma_log(t, gamma0=0.1, c=1):
#    return gamma0 / np.log(t + c)
#gamma = lambda t: gamma_log(t, gamma0=0.1, c=1)
#
## ejecutar descenso del gradiente
#theta, cost_history = gradient_descent(X, y, NITMAX, gamma)
#
## visualizar convergencia
#plt.plot(cost_history)
#plt.xlabel("Iteraciones")
#plt.ylabel("Costo")
#plt.title("Convergencia del descenso del gradiente")
#plt.grid()
#plt.show()