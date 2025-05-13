import pandas as pd
import networkx as nx
import random
import requests
import json
import time
from google.colab import files

# Configurazione dell'API di Mistral
MISTRAL_API_KEY = "INSERT API KEY"  

# Funzione per interrogare Mistral
def query_mistral(prompt, api_key, max_tokens=500, retries=3):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": max_tokens
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too Many Requests
                print(f"Errore 429 (tentativo {attempt + 1}/{retries}): Limite di richieste superato. Attendo 10 secondi...")
                time.sleep(10)  # Attendi 10 secondi prima di riprovare
            else:
                print(f"Errore API (tentativo {attempt + 1}/{retries}): {e}")
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Errore API (tentativo {attempt + 1}/{retries}): {e}")
            time.sleep(2)
    print(f"Impossibile ottenere risposta dopo {retries} tentativi")
    return ""

# Funzione per ottenere il punteggio Likert
def get_likert_score(concept1, concept2, api_key):
    prompt = f"""
    Valuta la forza della relazione semantica tra i concetti '{concept1}' e '{concept2}' su una scala Likert da 1 a 7, dove:
    1 = Nessuna relazione
    7 = Relazione molto forte
    Fornisci solo il numero del punteggio (es. 5).
    """

    response = query_mistral(prompt, api_key, max_tokens=10)
    # Pulizia della risposta: estrai il primo carattere non-spazio
    cleaned_response = ''.join(c for c in response if c.isdigit())[:1]  # Prendi solo il primo digit
    try:
        score = int(cleaned_response)
        if 1 <= score <= 7:
            return score
        else:
            print(f"Punteggio non valido per {concept1}-{concept2}: {score} (risposta originale: {response})")
            return None
    except ValueError:
        print(f"Risposta non valida per {concept1}-{concept2}: {response}")
        return None

# Passo 1: Carica il grafo dal file CSV
# Supponiamo che il CSV abbia colonne: source, target, weight
uploaded = files.upload()  # Carica il file CSV manualmente
csv_file = list(uploaded.keys())[0]  # Prendi il nome del file caricato
df = pd.read_csv(csv_file)

# Crea il grafo con NetworkX
G = nx.from_pandas_edgelist(df, source='source', target='target')

# Verifica il numero di nodi
print(f"Numero di nodi: {G.number_of_nodes()}")
print(f"Numero di archi: {G.number_of_edges()}")

# Passo 2: Seleziona un campione di coppie di concetti
num_samples = 300  # Numero di coppie da campionare
nodes = list(G.nodes())
concept_pairs = []

# Genera coppie casuali (senza ripetizioni)
while len(concept_pairs) < num_samples:
    pair = random.sample(nodes, 2)  # Seleziona due nodi casuali
    concept1, concept2 = sorted(pair)  # Ordina per evitare duplicati
    if (concept1, concept2) not in concept_pairs:
        concept_pairs.append((concept1, concept2))

# Passo 3: Richiedi i punteggi Likert per le coppie
results = []
for concept1, concept2 in concept_pairs:
    score = get_likert_score(concept1, concept2, MISTRAL_API_KEY)
    if score is not None:
        results.append({
            "Concept1": concept1,
            "Concept2": concept2,
            "LikertScore": score
        })
    # Aggiungi un ritardo per rispettare i limiti di quota dell'API
    time.sleep(2)  # 2 secondi di pausa

# Passo 4: Salva i risultati in un file CSV
results_df = pd.DataFrame(results)
output_file = "likert_scores.csv"
results_df.to_csv(output_file, index=False)
print(f"Risultati salvati in {output_file}")

# Scarica il file CSV
files.download(output_file)