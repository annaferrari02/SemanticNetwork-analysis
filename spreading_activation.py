import networkx as nx
import pandas as pd
from SpreadPy.Models.models import BaseSpreading
import numpy as np
from google.colab import files


def calculate_spreading(G, source, target, retention, timelist):
    initial_status = {node: 100 if node == source else 0 for node in G.nodes}
    model = BaseSpreading(G, retention=retention, decay=0, suppress=0)
    model.status = initial_status

    results = model.iteration_bunch(max(timelist))
    activations = []

    for t in timelist:
        activation = results[t-1]['status'].get(target, 0)
        activations.append(activation)

    return activations

print("Carica il file graph.csv (contiene gli archi del grafo)")
uploaded = files.upload()
graph_file = list(uploaded.keys())[0]
edges_df = pd.read_csv(graph_file, header=None)

print("Carica il file pairs.csv (contiene le coppie di parole)")
uploaded = files.upload()
pairs_file = list(uploaded.keys())[0]
pairs_df = pd.read_csv(pairs_file, header=None)

# Crea il grafo
G = nx.Graph()
for _, row in edges_df.iterrows():
    G.add_edge(row[0], row[1])

# Parametri
retention_rates = [0.1, 0.5, 0.9]  # Bassa, Media, Alta
timelist = [2,3,4,5, 10]  # Tempi di misurazione
results = []

# Per ogni coppia
for _, row in pairs_df.iterrows():
    word1, word2 = row[0], row[1]

    # Lista per i 9 valori medi (3 tempi × 3 condizioni)
    pair_results = []

    # Per ogni retention rate
    for retention in retention_rates:
        # Caso 1: Attiva word1, misura word2
        activations_1to2 = calculate_spreading(G, word1, word2, retention, timelist)

        # Caso 2: Attiva word2, misura word1
        activations_2to1 = calculate_spreading(G, word2, word1, retention, timelist)

        # Media delle attivazioni per ogni tempo
        for i in range(len(timelist)):
            avg_activation = (activations_1to2[i] + activations_2to1[i]) / 2
            pair_results.append(avg_activation)

    # Aggiungi risultato per la coppia
    results.append([word1, word2] + pair_results)

# Crea DataFrame con i risultati
columns = ['word1', 'word2'] + \
    [f'low_t{t}' for t in timelist] + \
    [f'med_t{t}' for t in timelist] + \
    [f'high_t{t}' for t in timelist]
results_df = pd.DataFrame(results, columns=columns)

# Salva i risultati e scarica il file
results_df.to_csv('spreading_results.csv', index=False)
files.download('spreading_results.csv')

print("Calcolo completato. Il file 'spreading_results.csv' è stato scaricato.")