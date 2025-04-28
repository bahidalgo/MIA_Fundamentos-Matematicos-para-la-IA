import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Carpeta para los frames
frames_dir = "perceptron_frames"
os.makedirs(frames_dir, exist_ok=True)

# Función escalón
def step_function(z):
    return 1 if z >= 0 else -1

# Perceptron_fit con visualización en cada época
def Perceptron_fit(X, y, nitmax, eta, visualize=False):
    X = np.array(X)
    y = np.array(y)
    n_samples, n_features = X.shape

    # Inicializa pesos incluyendo bias
    w = np.zeros(n_features + 1)
    frames = []

    if visualize:
        plot_decision_boundary(1, w, X, y, frames)

    for epoch in range(1, nitmax + 1):
        errors = 0
        for i in range(n_samples):
            xi_aug = np.append(X[i], 1)  # Agrega bias como entrada fija
            z = np.dot(w, xi_aug)
            y_pred = step_function(z)
            if y[i] != y_pred:
                w += eta * y[i] * xi_aug
                errors += 1
        if visualize:          
            plot_decision_boundary(epoch, w, X, y, frames)
        if errors == 0:
            break
    plot_decision_boundary(epoch, w, X, y, frames, is_final=True)
    return w, frames

# Predicción
def Perceptron_predict(w, X):
    X = np.array(X)
    return np.array([step_function(np.dot(np.append(xi, 1), w)) for xi in X])

# Evaluación
def Perceptron_score(w, X, y):
    preds = Perceptron_predict(w, X)
    return np.mean(preds == y)

# Visualización de la frontera de decisión
def plot_decision_boundary(epoch, w, X, y, frames_list, is_final=False):
    plt.figure(figsize=(6, 6))

    # Puntos
    for i in range(len(X)):
        color = 'blue' if y[i] == 1 else 'red'
        plt.scatter(X[i][0], X[i][1], color=color, s=100)

    # Frontera de decisión
    x_vals = np.array([0, 1])
    if w[1] != 0:
        y_vals = -(w[0] * x_vals + w[2]) / w[1]
        linestyle = '-' if is_final else '--'
        color = 'green' if is_final else 'black'
        plt.plot(x_vals, y_vals, color=color, linestyle=linestyle, linewidth=2,
                 label='Frontera Final' if is_final else f'Epoch {epoch}')
    else:
        linestyle = '-' if is_final else '--'
        color = 'green' if is_final else 'black'
        plt.axvline(x=-w[2]/w[0], color=color, linestyle=linestyle,
                    label='Frontera Final' if is_final else f'Epoch {epoch}')

    # Gráfico
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.title(f"Perceptrón - Epoch {epoch}")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.grid(True)
    plt.legend()
    frame_path = os.path.join(frames_dir, f"frame_{epoch:03d}.png")
    plt.savefig(frame_path)
    frames_list.append(frame_path)
    plt.close()


# Crear GIF a partir de los frames guardados
def generar_gif(frames, output_path="perceptron_training.gif", duration=800.0, freeze_last=3):
    with imageio.get_writer(output_path, mode='I', duration=duration, loop=0) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
        
        # Repetir el último frame algunas veces
        last_image = imageio.imread(frames[-1])
        for _ in range(freeze_last):
            writer.append_data(last_image)

    print(f"GIF generado en: {output_path}")


# Datos: compuerta OR con etiquetas {-1, 1}
X = [[0, 0], [0, 1], [1, 0], [1, 1], [1, 0.3], [0.1, 0.3], [0.2, 0.6]]
y = [-1, 1, 1, 1, 1, -1, 1]

# Entrenar y visualizar
w, frames = Perceptron_fit(X, y, nitmax=20, eta=0.2, visualize=True)

# Evaluar
acc = Perceptron_score(w, X, y)
print("Accuracy final:", acc)

# Generar GIF
generar_gif(frames, output_path="perceptron_training.gif")

