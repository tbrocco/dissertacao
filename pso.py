import numpy as np
from pyswarm import pso

# Função objetivo: calcula a sobreposição total entre os blocos
def objective_function(positions):
    num_blocks = len(positions)
    total_overlap = 0

    for i in range(num_blocks):
        for j in range(i + 1, num_blocks):
            x1, y1 = positions[i]
            x2, y2 = positions[j]
            w1, h1 = block_sizes[i]
            w2, h2 = block_sizes[j]

            overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
            total_overlap += overlap_x * overlap_y

    return total_overlap

# Tamanho dos blocos (largura x altura)
block_sizes = np.array([[2, 3], [4, 2], [3, 1]])

# Número de blocos
num_blocks = len(block_sizes)

# Limites das posições (defina conforme necessário)
lb = np.zeros((num_blocks, 2))
ub = np.ones((num_blocks, 2)) * 10

# Use PSO para otimizar as posições
best_position, best_score = pso(objective_function, lb, ub, swarmsize=30, maxiter=100)

print("Melhor posição encontrada:", best_position)
print("Melhor valor de sobreposição:", best_score)
