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
    axs[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
    axs[0, i].axis('off')
    axs[0, i].set_title("Original")

    axs[1, i].imshow(X_reconstruido[i].reshape(28, 28), cmap='gray')
    axs[1, i].axis('off')
    axs[1, i].set_title("PCA")

plt.tight_layout()
plt.savefig("mnist_pca32.png")











# Parte d)

def k_means_fit(X, Z0, NITERMAX):
    centroids = Z0.copy()
    objective_history = []

    for _ in range(NITERMAX):
        data = X[:, np.newaxis]
        distances = np.linalg.norm(data - centroids, axis=2)  # ← cálculo de distancias
        labels = np.argmin(distances, axis=1)                 # ← asignación de clústeres

        for k in range(centroids.shape[0]):
            if np.any(labels == k):
                centroids[k] = X[labels == k].mean(axis=0)    # ← actualización de centroides

        mu_z = centroids[labels]
        current_objective = np.mean((X - mu_z) ** 2)          # ← evaluación de función objetivo
        objective_history.append(current_objective)

    return centroids, labels, objective_history

def plot_representatives_and_samples(centroids, data, labels, n_samples=1):
    """
    Muestra cada centroide como imagen, junto con un ejemplo del conjunto de datos que fue asignado a él.
    """
    plt.figure(figsize=(14, 8))
    for i, centroid in enumerate(centroids):
        plt.subplot(4, 10, 2*i + 1)
        plt.imshow(centroid.reshape(28, 28), cmap='gray')
        plt.title(f'Centroide {i}', fontsize=8)
        plt.axis('off')

        assigned_points = data[labels == i]
        if len(assigned_points) > 0:
            plt.subplot(4, 10, 2*i + 2)
            plt.imshow(assigned_points[0].reshape(28, 28), cmap='gray')
            plt.title('Ejemplo', fontsize=8)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig("mnist_centroids_and_examples.png")



# Semilla reproducible
current_seed = 0
np.random.seed(current_seed)
k = 20
Z0 = X_reconstruido[np.random.choice(X_reconstruido.shape[0], k, replace=False)]

# Ejecutar algoritmo
max_iter = 6
results = {}

# Ejecutar k-means con 15 iteraciones
centroids, labels, objective_history = k_means_fit(X_reconstruido, Z0, 15)


# Gráfico de centroides y ejemplos
plot_representatives_and_samples(centroids, X_reconstruido, labels)
