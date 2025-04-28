import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import ssl
import urllib.request

ssl._create_default_https_context = ssl._create_unverified_context



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


def plot_objective(objective_history, data_label, figsize=(10, 5)):
    plt.figure(figsize=figsize)
    sns.set_style("darkgrid")
    ax = plt.gca()
    
    x_values = list(range(1, len(objective_history) + 1))  # Start from 1 instead of 0
    
    plt.plot(
        x_values,
        objective_history,
        marker='o',
        markersize=6,
        markerfacecolor='#FF6B6B',
        markeredgecolor='white',
        color='#4E79A7',
        linewidth=2
    )
    title = 'Convergencia de la función objetivo $J^{clust}$ para datos'
    plt.title(f'{title} {data_label}', fontsize=14, pad=20)
    plt.xlabel('Iteración', fontsize=12)
    plt.ylabel('Valor de la función objetivo', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"mnist_objective_curve_{data_label}.png")

# Gráfico de los clústeres

def plot_clusters_grid(data, results, current_seed):

    # Estilo visual
    sns.set_style("whitegrid")

    # Preparar variables
    n_clusters = len(next(iter(results.values()))[0])
    colors = sns.color_palette("husl", n_clusters)

    n_results = len(results)
    n_cols = min(3, n_results)
    n_rows = int(np.ceil(n_results / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, (n_iter, (centroids, labels)) in enumerate(results.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        for k in np.unique(labels):
            cluster_points = data[labels == k]
            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                color=colors[k],
                s=30,
                alpha=0.7,
                edgecolor='w',
                linewidth=0.5,
                label=f'Cluster {k}'
            )

        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c='black',
            marker='*',
            s=300,
            edgecolor='gold',
            linewidth=1.5,
            label='Centroids'
        )

        ax.set_title(f'Iteración {n_iter}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Feature 0', fontsize=11)
        ax.set_ylabel('Feature 1', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)

        for spine in ax.spines.values():
            spine.set_edgecolor('#DDDDDD')
            spine.set_linewidth(1.2)

    # Eliminar subplots vacíos
    for i in range(n_results, n_rows * n_cols):
        fig.delaxes(axes.flatten()[i])

    # Leyenda personalizada
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               label=f'Cluster {k}',
               markerfacecolor=colors[k],
               markersize=10)
        for k in range(n_clusters)
    ] + [
        Line2D([0], [0], marker='*', color='black',
               label='Centroids',
               markerfacecolor='gold',
               markersize=14,
               markeredgewidth=1.5)
    ]

    # Leyenda fuera del área del gráfico
    fig.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.9),
        ncol=min(n_clusters + 1, 6),
        frameon=True,
        fontsize=10,
        borderpad=0.5,
        labelspacing=0.8,
        columnspacing=1.2
    )


    plt.tight_layout(rect=[0, 0, 1, 0.84])
    #plt.show()
    plt.savefig(f"kmeans_seed{current_seed}.png")


# Cargar datos desde GitHub
url = "https://raw.githubusercontent.com/ManuelSanchezUribe/MIA_IMT3850_public/main/Semana1/datakmeans.csv"
df = pd.read_csv(url)
X = df.values

# Semilla reproducible
current_seed = 0
np.random.seed(current_seed)
k = 5
Z0 = X[np.random.choice(X.shape[0], k, replace=False)]

# Ejecutar algoritmo
max_iter = 6

results = {}

for iter_j in range(1, max_iter + 1):
    centroids, labels, objective_history = k_means_fit(X, Z0, NITERMAX=iter_j)
    
    results[iter_j] = (centroids, labels)



plot_clusters_grid(X, results, current_seed)





# Semilla reproducible
current_seed = 100
np.random.seed(current_seed)
k = 5
Z0 = X[np.random.choice(X.shape[0], k, replace=False)]

# Ejecutar algoritmo
max_iter = 6

results = {}

for iter_j in range(1, max_iter + 1):
    centroids, labels, objective_history = k_means_fit(X, Z0, NITERMAX=iter_j)
    
    results[iter_j] = (centroids, labels)



plot_clusters_grid(X, results, current_seed)

# Gráfico de convergencia de la función objetivo
plot_objective(objective_history, "datakmeans")



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




from tensorflow.keras.datasets import mnist

# Cargar MNIST desde Keras
(X_train, y_train), (_, _) = mnist.load_data()

# Aplanar imágenes y normalizar
X = X_train.reshape((X_train.shape[0], -1)).astype(np.float32) / 255.0

# Seleccionar una muestra pequeña
sample_size = 10000
X = X[:sample_size]

# Semilla reproducible
current_seed = 0
np.random.seed(current_seed)
k = 20
Z0 = X[np.random.choice(X.shape[0], k, replace=False)]

# Ejecutar algoritmo
max_iter = 6
results = {}

# Ejecutar k-means con 15 iteraciones
centroids, labels, objective_history = k_means_fit(X, Z0, 15)

# Gráfico de convergencia de la función objetivo
plot_objective(objective_history, "MNIST")

# Gráfico de centroides y ejemplos
plot_representatives_and_samples(centroids, X, labels)







