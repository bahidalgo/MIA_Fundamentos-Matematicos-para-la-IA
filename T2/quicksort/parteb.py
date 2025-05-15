import random
def random_quicksort(S):
    # Si S no tiene elementos o solo un elemento,
    # se retorna S
    if len(S) <= 1:
        return S, 0
    # Escoger un elemento x de S
    pivot_index = random.randint(0, len(S) - 1)
    x = S[pivot_index]
    
    comparisons = 0
    # Se crea nuevas sublistas
    S1 = []
    S2 = []
    # Se compara cada elemento de S con x
    for i in range(len(S)):
        if i != pivot_index:
            comparisons += 1
            if S[i] < x:
                # S1 tiene elementos menores a x
                S1.append(S[i])
            else:
                # S2 tiene elementos mayores a x
                S2.append(S[i])
    # Se ordena S1 usando Quicksort
    sorted_S1, comp1 = random_quicksort(S1)
    # Se ordena S2 usando Quicksort
    sorted_S2, comp2 = random_quicksort(S2)
    # Al finalizar la recurrencia, se obtiene el
    # número de comparaciones
    total_comparisons = comparisons + comp1 + comp2
    return sorted_S1 + [x] + sorted_S2, total_comparisons

S = [0, 5, 4, 1, 7, 6, 3, 2, 8, 9]
sorted_S, total_comparisons = random_quicksort(S.copy())

print(f"Lista inicial: {S}")
print(f"Lista ordenada: {sorted_S}")
print(f"Comparaciones realizadas: {total_comparisons}")    

def promedio_comparaciones(n, repeticiones=100):
    # Acumulador para todas las comparaciones
    total_comparaciones = 0  

    for _ in range(repeticiones):
        # Genera una lista aleatoria de n elementos distintos
        lista_aleatoria = random.sample(range(n * 10), n)

        # Aplica el algoritmo Random Quicksort y captura el número de comparaciones
        _, comparaciones = random_quicksort(lista_aleatoria)

        # Suma las comparaciones a la cuenta total
        total_comparaciones += comparaciones

    # Calcula el promedio dividiendo por el número de repeticiones
    promedio = total_comparaciones / repeticiones
    return promedio

# Ejemplo de uso para n fijo
n_fijo = 100  # Tamaño de la lista fija
prom = promedio_comparaciones(n_fijo)

# Mostrar resultado
print(f"Promedio de comparaciones para n = {n_fijo}: {prom:.2f}")