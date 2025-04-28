import numpy as np

A = np.array([
    [4, 1, 0],
    [1, 4, 1],
    [0, 1, 4]
])

# Calcular norma de Frobenius directamente
frobenius = np.linalg.norm(A, ord='fro')
print(f"Norma de Frobenius: {frobenius:.4f}")

# Calcular valores singulares
sigma = np.linalg.svd(A, compute_uv=False)
print("Valores singulares:", sigma)

# Verificar identidad
print("Suma de cuadrados de los σ_i:", np.sum(sigma**2))
