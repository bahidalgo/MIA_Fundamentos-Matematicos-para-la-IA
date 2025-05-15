import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from partea import gradient_descent
from parteb import stochastic_gradient_descent

# --------------------------------------------
# Cargar datos desde CSV
# --------------------------------------------
data = pd.read_csv("datos_lineales.csv")
X_raw = data[['x']].values
y = data['y'].values

# Agregar columna de unos para el término independiente (bias)
X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

# --------------------------------------------
# Definición de funciones de tasa de aprendizaje gamma(t)
# --------------------------------------------

def gamma_log(t, gamma0=0.1, c=1):
    """Tasa de aprendizaje logarítmica decreciente"""
    return gamma0 / np.log(t + c)

def gamma_clase(t, a=1, k0=1):
    """Heurística vista en clases: tipo 1 / (a * (t + k0))"""
    return 1 / (a * (t + k0))

# --------------------------------------------
# Ejecutar SGD múltiples veces y promediar costos por iteración
# --------------------------------------------
def promedio_costos_sgd(X, y, NITMAX, gamma_fn, reps=100):
    """
    Ejecuta SGD 'reps' veces con gamma(t) y promedia el costo en cada iteración.
    """
    costos_todas = []
    for _ in range(reps):
        _, costos = stochastic_gradient_descent(X, y, NITMAX, gamma_fn)
        costos_todas.append(costos)
    return np.mean(costos_todas, axis=0)

# --------------------------------------------
# Parámetros generales
# --------------------------------------------
NITMAX = 1000
gamma_fn_log = lambda t: gamma_log(t, gamma0=0.1, c=1)
gamma_fn_clase = lambda t: gamma_clase(t, a=1, k0=1)

# --------------------------------------------
# Ejecutar algoritmos
# --------------------------------------------

# Estocástico con tasa logarítmica
costos_log = promedio_costos_sgd(X, y, NITMAX, gamma_fn=gamma_fn_log)

# Estocástico con tasa tipo clase
costos_clase = promedio_costos_sgd(X, y, NITMAX, gamma_fn=gamma_fn_clase)

# Ejecutar gradient descent con funciones gamma(t)
_, costos_det_log = gradient_descent(X, y, NITMAX, gamma=gamma_fn_log)
_, costos_det_clase = gradient_descent(X, y, NITMAX, gamma=gamma_fn_clase)

# --------------------------------------------
# Graficar convergencia promedio del costo
# --------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(costos_log, label='SGD γ_log', alpha=0.8)
plt.plot(costos_clase, label='SGD γ_clases', alpha=0.8)
plt.plot(costos_det_log, label='GD γ_log', alpha=0.8, linestyle='--', linewidth=2)
plt.plot(costos_det_clase, label='GD γ_clases', alpha=0.8, linestyle='--', linewidth=2)

plt.xlabel("Iteraciones")
plt.ylabel("Costo promedio")
plt.title("Convergencia promedio del costo en 100 ejecuciones")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
