import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from partea import gradient_descent
from parteb import stochastic_gradient_descent

# Cargar datos cuadráticos desde CSV
data = pd.read_csv("datos_cuadraticos.csv")
X_raw = data[['x']].values
y = data['y'].values

# Construir matriz de diseño: [1, x]
X = np.hstack([np.ones((X_raw.shape[0], 1)), X_raw])

# Definir funciones de tasa de aprendizaje
def gamma_log(t, gamma0=0.1, c=1):
    return gamma0 / np.log(t + c)

def gamma_clase(t, a=1, k0=1):
    return 1 / (a * (t + k0))

# Función para promedio de costos en SGD
def promedio_costos_sgd(X, y, NITMAX, gamma_fn, reps=100):
    costos_todas = []
    for _ in range(reps):
        _, costos = stochastic_gradient_descent(X, y, NITMAX, gamma_fn)
        costos_todas.append(costos)
    return np.mean(costos_todas, axis=0)

# Parámetros
NITMAX = 1000
gamma_fn_log = lambda t: gamma_log(t, gamma0=0.1, c=1)
gamma_fn_clase = lambda t: gamma_clase(t, a=1, k0=1)

# Ejecutar SGD con ambas tasas
costos_log = promedio_costos_sgd(X, y, NITMAX, gamma_fn=gamma_fn_log)
costos_clase = promedio_costos_sgd(X, y, NITMAX, gamma_fn=gamma_fn_clase)

# Ejecutar GD con ambas tasas (1 vez cada uno)
_, costos_det_log = gradient_descent(X, y, NITMAX, gamma=gamma_fn_log)
_, costos_det_clase = gradient_descent(X, y, NITMAX, gamma=gamma_fn_clase)

# Graficar comparación de tasas: SGD vs GD
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# SUBPLOT 1: SGD
axs[0].plot(costos_log, label='SGD γ_log', alpha=0.9)
axs[0].plot(costos_clase, label='SGD γ_clases', alpha=0.9)
axs[0].set_xlabel("Iteraciones")
axs[0].set_ylabel("Costo promedio")
axs[0].set_title("SGD en datos cuadráticos")
axs[0].legend()
axs[0].grid(True)
axs[0].set_ylim(0.04, 0.05)

# SUBPLOT 2: GD
axs[1].plot(costos_det_log, label='GD γ_log', alpha=0.9)
axs[1].plot(costos_det_clase, label='GD γ_clases', alpha=0.9)
axs[1].set_xlabel("Iteraciones")
axs[1].set_ylabel("Costo promedio")
axs[1].set_title("GD en datos cuadráticos")
axs[1].legend()
axs[1].grid(True)
axs[1].set_ylim(0.04, 0.05)

plt.tight_layout()
plt.savefig("comparacion_tasas_datos_cuadraticos.png", dpi=300)
plt.show()
