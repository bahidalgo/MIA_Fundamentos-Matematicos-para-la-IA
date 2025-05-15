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



