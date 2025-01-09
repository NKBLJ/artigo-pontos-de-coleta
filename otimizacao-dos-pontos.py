import numpy as np
from scipy.optimize import differential_evolution
import pandas as pd

# Importação dos dados
dados_bairros = pd.read_csv("dados/bairros-agrupados-local.csv", sep=";", encoding='latin')
dados_coleta = pd.read_csv("dados/pontos-coleta-com-coords.csv", sep=";", encoding='latin')

# Pontos dos bairros
bairros = np.array(dados_bairros[['latitude', 'longitude']].values)
populacoes = np.array(dados_bairros[['População']].values)

# Pontos fixos
pontos_fixos = np.array(dados_coleta.loc[dados_coleta['tipo'] == 'fixo', ['latitude', 'longitude']])

# Número de novos pontos de coleta
k = 4  # Número de novos pontos

# Calcular os limites (bounds)
x_min, x_max = bairros[:, 0].min(), bairros[:, 0].max()
y_min, y_max = bairros[:, 1].min(), bairros[:, 1].max()
bounds = [(x_min, x_max), (y_min, y_max)] * k  # Limites para cada ponto

# Função de custo com pesos
def funcao_objetivo(pontos):
    pontos = pontos.reshape((k, 2))  # Reformata para coordenadas (x, y)
    todos_pontos = np.vstack([pontos_fixos, pontos])  # Inclui os pontos fixos
    custo_total = 0
    for bairro, pop in zip(bairros, populacoes):
        distancias = np.linalg.norm(bairro - todos_pontos, axis=1)  # Distâncias a todos os pontos
        custo_total += pop * np.min(distancias)  # Soma ponderada pela população
    return custo_total

# Otimização com Differential Evolution
resultado = differential_evolution(funcao_objetivo, bounds)

# Resultado final
pontos_otimizados = resultado.x.reshape((k, 2))
print("Novo ponto de coleta otimizado:")
print(pontos_otimizados)

# Todos os pontos
todos_pontos_finais = np.vstack([pontos_fixos, pontos_otimizados])
print("Todos os pontos de coleta (fixos e novos):")
print(todos_pontos_finais)

# Custo total após a otimização
custo_final = resultado.fun
print(f"Custo total otimizado: {custo_final}")
