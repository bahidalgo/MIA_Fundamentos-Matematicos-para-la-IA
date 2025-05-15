import numpy as np

def stochastic_gradient_descent(X, y, NITMAX, gamma):
    """
    Algoritmo de descenso de gradiente estocástico para minimizar:
        J(θ) = (1/2N) * sum((x_i^T θ - y_i)^2)
    """
    N = X.shape[0]
    theta = np.zeros((2,))
    history = []

    for t in range(1, NITMAX + 1):
        # Paso 1: seleccionar un dato aleatorio
        i = np.random.randint(N)
        xi = X[i]
        yi = y[i]

        # Paso 2: calcular gradiente estocástico
        y_pred_i = xi @ theta
        error_i = y_pred_i - yi
        grad_i = error_i * xi

        # Paso 3: actualizar theta con tasa dependiente de t
        theta = theta - gamma(t) * grad_i

        # Paso 4: registrar costo cada 10 iteraciones
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
## Ejecutar SGD
#theta_sgd, cost_history_sgd = stochastic_gradient_descent(X, y, NITMAX, gamma)
#
#
#plt.plot(np.arange(len(cost_history_sgd))*100, cost_history_sgd, label="SGD cost")
#plt.xlabel("Iteraciones")
#plt.ylabel("Costo")
#plt.title("Convergencia del descenso de gradiente estocástico")
#plt.grid()
#plt.legend()
#plt.show()