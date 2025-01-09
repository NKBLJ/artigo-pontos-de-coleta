import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

# Carregar os dados
df = pd.read_csv('dados/ibge-2022-com-coordenadas.csv', sep=';', encoding='latin')

# Função para calcular o centro de gravidade ponderado
def centro_de_gravidade(group):
    total_pop = group['População'].sum()
    lat_cg = np.sum(group['latitude'] * group['População']) / total_pop
    lon_cg = np.sum(group['longitude'] * group['População']) / total_pop
    return pd.Series({'População-somada': total_pop, 'latitude': lat_cg, 'longitude': lon_cg})

# Calcular matriz de distância entre bairros (usando latitude e longitude)
coords = df[['latitude', 'longitude']].values
dist_matrix = squareform(pdist(coords, metric='euclidean'))

# Clustering hierárquico para agrupar bairros próximos
threshold = 0.175  # Defina o limite de distância para agrupar (ajuste conforme necessário)
clusters = fcluster(linkage(dist_matrix, method='average'), t=threshold, criterion='distance')
df['Cluster'] = clusters

# Agrupar por cluster e calcular os resultados
result = df.groupby('Cluster').apply(
    lambda group: pd.Series({
        'Bairro': '-'.join(group['Bairro']),
        'População': group['População'].sum(),
        'latitude': (group['latitude'] * group['População']).sum() / group['População'].sum(),
        'longitude': (group['longitude'] * group['População']).sum() / group['População'].sum(),
    })
).reset_index(drop=True)

# Salvar o resultado em um novo CSV
result.to_csv('bairros-agrupados-local.csv', sep=';', encoding='latin', index=False)

# Exibir o resultado
print(result)
