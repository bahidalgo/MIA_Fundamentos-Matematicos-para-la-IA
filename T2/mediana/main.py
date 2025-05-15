# Reimportar lo necesario tras reset
import numpy as np
import matplotlib.pyplot as plt
import math

def mediana_aleatoria_con_fallos(S):
    """
    Algoritmo de mediana aleatoria visto en clases.
    Repite hasta que encuentra la mediana, registrando cuántos intentos fallidos hubo antes del éxito.
    """
    n = len(S)
    fallos = 0

    while True:
        t = int(n**(3/4))  # Tamaño de muestra

        # Paso 1: muestreo aleatorio con reemplazo
        R = np.random.choice(S, size=t, replace=True)

        # Paso 2: ordenar el conjunto R
        R.sort()

        # Paso 3: calcular los índices de los cuantiles internos
        delta = math.sqrt(n)
        i_d = max(0, int(t / 2 - delta))    # índice inferior
        i_u = min(t - 1, int(t / 2 + delta))  # índice superior

        d = R[i_d]
        u = R[i_u]

        # Paso 4: clasificar elementos en tres grupos
        C = [x for x in S if d <= x <= u]
        l_d = [x for x in S if x < d]
        l_u = [x for x in S if x > u]

        # Paso 5: verificar si la mediana está en C
        if len(C) > n / 2 and len(l_d) <= n / 2 and len(l_u) <= n / 2:
            C.sort()
            i = n // 2 - len(l_d)
            return C[i], fallos  # éxito
        else:
            fallos += 1

# ------------------------------
# Experimento visual de convergencia
# ------------------------------
def visualizar_fallos_como_conteo(n=100, repeticiones=1000):
    """
    Ejecuta el algoritmo muchas veces, y registra cuántos fallos se producen antes de acertar.
    Luego grafica el promedio acumulado y la cota teórica n^{-1/4}.
    """
    fallos_por_ejecucion = []

    for _ in range(repeticiones):
        S = np.random.randint(0, 1000, size=n)
        _, fallos = mediana_aleatoria_con_fallos(S)
        fallos_por_ejecucion.append(fallos)

    # Promedio acumulado de fallos
    promedio_acumulado = np.cumsum(fallos_por_ejecucion) / np.arange(1, repeticiones + 1)
    cota_teorica = n ** (-1/4)

    # Graficar resultados
    plt.figure(figsize=(10, 5))
    plt.plot(promedio_acumulado, label="Promedio acumulado de fallos")
    plt.axhline(cota_teorica, color="red", linestyle="--", label=f"Cota teórica: $n^{{-1/4}}$ ≈ {cota_teorica:.4f}")
    plt.xlabel("Número de ejecuciones")
    plt.ylabel("Fallos promedio antes del éxito")
    plt.title(f"Convergencia de intentos fallidos promedio vs cota teórica con n={n}")
    plt.ylim(0, max(0.5, np.max(promedio_acumulado) + 0.05))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = "Random_median.jpg"
    plt.savefig(filename)

    return promedio_acumulado[-1], cota_teorica

# Ejecutar visualización y mostrar últimos valores
promedio_final, cota = visualizar_fallos_como_conteo()
print(promedio_final, cota)
