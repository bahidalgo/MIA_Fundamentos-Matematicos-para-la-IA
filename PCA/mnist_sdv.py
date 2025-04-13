from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Parte a)
# Cargar MNIST desde Keras
(X_train, y_train), (_, _) = mnist.load_data()

# Aplanar imágenes y normalizar
X = X_train.reshape((X_train.shape[0], -1)).astype(np.float32) / 255.0

# Seleccionar una muestra pequeña
sample_size = 10000
X = X[:sample_size]

# Centrar los datos (restar la media por columna)
X_centered = X - np.mean(X, axis=0)

# Aplicar SVD
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

# Mostrar resultados
print("Valores singulares (primeros 10):")
print(S[:10])


# Parte b)
# Elegimos la mitad de la dimensión original
k = int(X_centered.shape[1]/2)

# Proyección sobre los primeros k vectores singulares
X_reducido = X_centered @ Vt[:k].T

# Confirmar dimensiones
print("Forma original:", X_centered.shape)
print("Forma reducida:", X_reducido.shape)


# Parte c)

# Reconstrucción aproximada de los datos
X_reconstruido = X_reducido @ Vt[:k]

# Graficar las primeras 10 imágenes originales y sus proyecciones
fig, axs = plt.subplots(2, 10, figsize=(15, 4))
for i in range(10):
    axs[0, i].imshow(X_centered[i].reshape(28, 28), cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title("Original")

    axs[1, i].imshow(X_reconstruido[i].reshape(28, 28), cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title("PCA")

plt.tight_layout()
plt.savefig("mnist_pca32.png")
