import numpy as np

def Perceptron_fit(X, y, nitmax, eta):
    # Cada x ∈ ℝⁿ⁻¹ debe ser aumentado con una entrada unitaria.
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])

    # Inicializamos w = 0 ∈ ℝⁿ
    w = np.zeros(X_aug.shape[1])
    
    # Guardamos el historial de w
    historia = [w.copy()] 

    # Bucle principal: hasta nitmax o convergencia
    for _ in range(nitmax):
        actualizado = False
        for i in range(X_aug.shape[0]):
            # Calcular ŷ = sgn(xᵢᵀ w)
            y_pred = np.sign(np.dot(X_aug[i], w))
            if y_pred == 0:
                y_pred = -1  # por convención

            # Si está mal clasificado, actualizar pesos
            if y_pred != y[i]:
                # Aplicamos la regla de aprendizaje:
                # w(k+1) = w(k) − η⋅y⋅xi,
                w -= eta * y[i] * X_aug[i]
                actualizado = True
        historia.append(w.copy())
        if not actualizado:
            break  # convergió

    return w, historia


def Perceptron_predict(w, X):
    # Aumentar con entrada unitaria (sesgo)
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    
    # Aplicar la regla: ŷ = sgn(Xw)
    y_pred = np.sign(X_aug @ w)
    y_pred[y_pred == 0] = -1 
    return y_pred


def Perceptron_score(w, X, y):
    # Hacemos uso de la función creada en b)
    y_pred = Perceptron_predict(w, X)
    # Simplemente consideraremos el porcentaje
    # de detecciones correctas.
    score = (y_pred == y).mean() * 100
    return score





import imageio.v2 as imageio
import os
import pandas as pd
import matplotlib.pyplot as plt

def cargar_y_preparar_datos(nombre_archivo):
    # Leer el archivo CSV
    df = pd.read_csv(nombre_archivo)

    # Extraer variables y etiquetas
    X = df[['x', 'y']].values
    y = df['label'].values

    # Convertir etiquetas de {0, 1} a {-1, 1}
    y = 2 * y - 1

    return X, y


def generar_gif_hiperplano(X, y, historia, nombre_gif='perceptron.gif', duration=500, freeze_last=3):

    filenames = []

    for epoca, w in enumerate(historia):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', label='Datos')

        x1_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200)
        if w[1] != 0:
            x2_vals = -(w[0] * x1_vals + w[2]) / w[1]
            ax.plot(x1_vals, x2_vals, 'k--', label='Hiperplano')
        else:
            x_vert = -w[2] / w[0]
            ax.axvline(x=x_vert, color='k', linestyle='--', label='Hiperplano')

        ax.set_title(f'Época {epoca}')
        ax.set_xlim([-8, 8])
        ax.set_ylim([-8, 8])
        ax.grid(True)

        filename = f'_frame_{epoca}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

    with imageio.get_writer(nombre_gif, mode='I', duration=duration, loop=0) as writer:
        for i, filename in enumerate(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)
        
        # Repetir el último frame algunas veces
        #last_image = imageio.imread(filename)
        for _ in range(freeze_last):
            writer.append_data(image)


    # Eliminar imágenes temporales
    for filename in filenames:
        os.remove(filename)

    print(f"GIF generado: {nombre_gif}")





# --- Para datos1.csv ---
X1, y1 = cargar_y_preparar_datos('datos1.csv')
w1, history1 = Perceptron_fit(X1, y1, nitmax=20, eta=-0.01)
generar_gif_hiperplano(X1, y1, history1, nombre_gif='perceptron_datos1.gif')

# --- Para datos2.csv ---
X2, y2 = cargar_y_preparar_datos('datos2.csv')
w2, history2 = Perceptron_fit(X2, y2, nitmax=20, eta=-0.01)
generar_gif_hiperplano(X2, y2, history2, nombre_gif='perceptron_datos2.gif')







# Paso 0: Primero veremos cómo le fue en términos de score para el primer dataset.
score = Perceptron_score(w1, X1, y1)
print(f"Score sobre datos1.csv: {score:.2f}%")


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Paso 1: Generar 10 nuevos datos y etiquetas
np.random.seed(18)
new_data = np.random.random(size=(10, 2))
new_data_labels = np.random.randint(2, size=10)
new_data_labels = 2 * new_data_labels - 1  # Convertir a {-1, 1}

# Paso 2: Predecir las clases con el perceptrón entrenado en datos1
predicciones = Perceptron_predict(w1, new_data)

# Paso 3: Calcular el score
score = Perceptron_score(w1, new_data, new_data_labels)
print(f"Score sobre nuevos datos aleatorios: {score:.2f}%")

# Paso 4: Graficar conjunto original + nuevos datos
plt.figure(figsize=(8, 6))

# Datos originales
plt.scatter(X1[:, 0], X1[:, 1], c=y1, cmap='bwr', edgecolors='k', alpha=0.5, marker='o')

# Nuevos datos clasificados
plt.scatter(new_data[:, 0], new_data[:, 1], c=new_data_labels, cmap='bwr', edgecolors='k', s=100, marker='P')

# Hiperplano
x1_vals = np.linspace(-8, 8, 200)
if w1[1] != 0:
    x2_vals = -(w1[0] * x1_vals + w1[2]) / w1[1]
    plt.plot(x1_vals, x2_vals, 'k--')
else:
    x_vert = -w1[2] / w1[0]
    plt.axvline(x=x_vert, color='k', linestyle='--')

# Crear leyenda personalizada (sin colores de clase)
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Datos originales',
           markerfacecolor='none', markeredgecolor='black', markersize=8),
    Line2D([0], [0], marker='P', color='w', label='Nuevos datos',
           markerfacecolor='none', markeredgecolor='black', markersize=10),
    Line2D([0], [0], linestyle='--', color='black', label='Hiperplano')
]

plt.legend(handles=legend_elements, loc='upper right')

filename = "perceptron_ajuste_datos_nuevos.jpg"
# Ajustes visuales
plt.xlabel("x")
plt.ylabel("y")
plt.xlim([-6, 6])
plt.ylim([-6, 6])

plt.grid(True)
plt.savefig(filename)


