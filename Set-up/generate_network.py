import networkx as nx
import requests
import json
import time
import matplotlib.pyplot as plt
import random

# CONFIG 
api_key = "INSERT API KEY"
seed_words = [
    "apple", "fruit", "dog", "animal", "rose", "flower",
    "hammer", "tool", "car", "vehicle",
    "shirt", "clothing", "oak", "tree",
    "violin", "musical instrument", "red", "color"
]
# INITIAL PROMPT
initial_prompt_template = """
Create a dense semantic network that includes the word '{seed_word}' and reflects natural associations between concepts.

STRICT CONSTRAINTS:
- Generate EXACTLY 15 concept pairs.
- Use EXACTLY 7-9 distinct nodes (no more, no less).
- Include '{seed_word}' in at least 5 of the 15 pairs.

FORMAT:
- Each line must contain exactly one pair in the format "concept1, concept2"
- DO NOT include numbers, introductions, or explanations.
- DO NOT exceed 9 total concepts.

CORRECT FORMAT EXAMPLE:
apple, fruit
apple, tree
fruit, sweet
tree, nature
apple, sweet
etc.

REQUIRED WORD: {seed_word}

"""

#PROMPT PER ESPANSIONE
expansion_prompt_template = """
Expand the semantic network using the following concepts: {existing_concepts}.

STRICT CONSTRAINTS:
- Generate EXACTLY 15 ADDITIONAL pairs.
- Use ONLY the existing concepts — do NOT introduce any new terms.
- Each pair must be in the format "concept1, concept2"

CONCEPTS TO USE: {existing_concepts}

CORRECT FORMAT EXAMPLE:
concept1, concept2
concept3, concept4
etc.
"""
#QUERY MISTRAL AI 
def query_mistral(prompt, api_key, max_tokens=1000, retries=3):
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
        except requests.exceptions.RequestException as e:
            print(f"Errore API (tentativo {attempt + 1}/{retries}): {e}")
            time.sleep(2)
    print(f"Impossibile ottenere risposta dopo {retries} tentativi")
    return ""

def suggest_semantic_connections(node1, node2, api_key):
    prompt = f"""
    Is there a meaningful semantic relationship between '{node1}' and '{node2}'?

    Consider relationships such as:
    - Category membership
    - Use or function
    - Shared contexts
    - Cultural associations
    - Part-whole relationships

    Respond only with 'YES' or 'NO'.

    """
    response = query_mistral(prompt, api_key, max_tokens=10)
    return "YES" in response.upper()




def parse_edges(response_text, seed_word, expected_edges, min_nodes, max_nodes, phase="initial"):
    edges = []
    nodes = set()
    lines = response_text.strip().split("\n")
    seed_word_count = 0

    # Prima passata: priorità alle coppie con seed_word
    for line in lines:
        if "," in line:
            try:
                source, target = line.split(",", 1)
                source, target = source.strip(), target.strip()
                if source == seed_word or target == seed_word:
                    edges.append((source, target))
                    nodes.add(source)
                    nodes.add(target)
                    seed_word_count += 1
            except Exception as e:
                print(f"Errore nel parsing della riga: {line}")

    # Seconda passata: altre coppie fino al limite di nodi
    for line in lines:
        if "," in line and len(edges) < expected_edges:
            try:
                source, target = line.split(",", 1)
                source, target = source.strip(), target.strip()
                if source != seed_word and target != seed_word:
                    # Controlla se l'aggiunta supererebbe il limite di nodi
                    new_nodes = set([source, target]) - nodes
                    if len(nodes) + len(new_nodes) <= max_nodes:
                        edges.append((source, target))
                        nodes.add(source)
                        nodes.add(target)
            except Exception as e:
                continue

    # Verifiche e adattamenti
    if phase == "initial" and seed_word_count < 5:
        print(f" {phase.capitalize()} - Solo {seed_word_count} archi includono '{seed_word}'")

    if len(nodes) < min_nodes:
        print(f" {phase.capitalize()} - Generati solo {len(nodes)} nodi, inferiore al minimo di {min_nodes}")

    if len(nodes) > max_nodes:
        print(f" {phase.capitalize()} - Generati {len(nodes)} nodi, superiore al massimo di {max_nodes}")

    if len(edges) != expected_edges:
        print(f" {phase.capitalize()} - Generati {len(edges)} archi invece di {expected_edges}")

    return edges, nodes


#Optimize density 
def optimize_density(G, target_density=0.15, api_key=None, max_attempts=100):
    """Optimize the density of the semantic graph by adding only semantically valid edges.

      STRICT CONSTRAINTS:
      - Use only the existing nodes: {existing_nodes}
      - Add only edges that reflect a meaningful semantic relationship
      - Relationships must meet at least one of the following criteria:
          • Category membership
          • Use or function
          • Shared context or co-occurrence
          • Cultural or common-sense association
          • Part-whole structure

      FORMAT:
      - Output only NEW edges in the format "node1, node2"
      - Do NOT repeat existing edges
      - Do NOT introduce new nodes
      - Do NOT include explanations

      EXAMPLE OF CORRECT FORMAT:
      apple, fruit
      car, vehicle
      rose, flower """
    current_density = nx.density(G) if G.number_of_nodes() > 1 else 0
    attempts = 0

    print(f" Ottimizzazione densità: iniziale {current_density:.3f}, target {target_density:.3f}")

    non_edges = list(nx.non_edges(G))
    random.shuffle(non_edges)

    for source, target in non_edges:
        if current_density >= target_density or attempts >= max_attempts:
            break

        if api_key is not None and suggest_semantic_connections(source, target, api_key):
            time.sleep(1.5)  # attesa per evitare rate limit
            G.add_edge(source, target)
            print(f"➕ Aggiunto arco semantico: {source} - {target}")
            current_density = nx.density(G)
        attempts += 1

    print(f"Ottimizzazione completata: densità finale = {current_density:.3f}")
    return G


#MAIN FUNCTION
def create_semantic_network(target_density=0.15):
    # Inizializza il grafo come unione di sottoreti per ogni seed word
    master_graph = nx.Graph()
    seed_graphs = {}

    for seed_word in seed_words:
        start_time = time.time()
        print(f"\n🔍 Analisi per la parola: {seed_word}")

        # Crea un grafo specifico per questo seed word
        G = nx.Graph()

        # Fase 1: Rete iniziale
        initial_prompt = initial_prompt_template.format(seed_word=seed_word)
        initial_response = query_mistral(initial_prompt, api_key, max_tokens=200)
        initial_edges, initial_nodes = parse_edges(initial_response, seed_word,
                                                 expected_edges=15,
                                                 min_nodes=7, max_nodes=9,
                                                 phase="initial")

        # Aggiungi archi iniziali al grafo
        for edge in initial_edges:
            G.add_edge(*edge)

        # Ottimizza la densità del sotto-grafo
        if nx.density(G) < target_density:
            # Fase 2: Espansione se necessario
            existing_concepts = ", ".join(G.nodes())
            expansion_prompt = expansion_prompt_template.format(existing_concepts=existing_concepts)
            expansion_response = query_mistral(expansion_prompt, api_key, max_tokens=300)
            expansion_edges, _ = parse_edges(expansion_response, seed_word,
                                           expected_edges=15,
                                           min_nodes=len(G.nodes()),
                                           max_nodes=len(G.nodes()),
                                           phase="expansion")

            # Aggiungi archi di espansione al grafo
            for edge in expansion_edges:
                G.add_edge(*edge)

        # Ottimizza la densità finale per questo seed word
        G = optimize_density(G, target_density, api_key=api_key, max_attempts=30)

        # Salva il grafo per questo seed word
        seed_graphs[seed_word] = G

        # Aggiungi nodi e archi al grafo master
        for node in G.nodes():
            if node not in master_graph:
                master_graph.add_node(node)

        for edge in G.edges():
            master_graph.add_edge(*edge)

        # Statistiche
        density = nx.density(G)
        degree_avg = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        print(f" Densità del grafo per '{seed_word}': {density:.3f} (Obiettivo: {target_density})")
        print(f" Grado medio dei nodi: {degree_avg:.2f}")
        print(f" Numero di nodi: {G.number_of_nodes()}")
        print(f" Numero di archi: {G.number_of_edges()}")

        duration = time.time() - start_time
        print(f"⏱ Tempo per '{seed_word}': {duration:.2f} secondi")

        time.sleep(1)  # Pausa per non sovraccaricare l'API

    # Ottimizza la densità del grafo complessivo
    master_graph = optimize_density(master_graph, target_density, api_key=api_key, max_attempts=300)

    # Statistiche finali
    final_density = nx.density(master_graph)
    final_degree_avg = sum(dict(master_graph.degree()).values()) / master_graph.number_of_nodes() if master_graph.number_of_nodes() > 0 else 0
    print(f"\n RISULTATO FINALE:")
    print(f" Densità finale del grafo: {final_density:.3f} (Obiettivo: {target_density})")
    print(f" Grado medio dei nodi: {final_degree_avg:.2f}")
    print(f" Numero di nodi totali: {master_graph.number_of_nodes()}")
    print(f" Numero di archi totali: {master_graph.number_of_edges()}")

    return master_graph, seed_graphs

#VISUALIZE GRAPH 
def draw_graph(G, title="Rete semantica", filename=None):
    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G, seed=42, k=0.2)  # k regola la distanza tra i nodi

    # Colora i nodi di seed in modo diverso
    node_colors = []
    for node in G.nodes():
        if node in seed_words:
            node_colors.append('lightcoral')
        else:
            node_colors.append('skyblue')

    # Dimensione dei nodi basata sul grado
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes()]

    # Disegna il grafo
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title(f"{title} - Densità: {nx.density(G):.3f}")
    plt.axis('off')

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

#esecution
if __name__ == "__main__":
    master_graph, seed_graphs = create_semantic_network(target_density=0.15)

    # Disegna il grafo completo
    draw_graph(master_graph, title="Rete semantica completa", filename="rete_semantica_completa.png")

    # Opzionale: disegna anche i sottografi per ogni seed word
    #for seed_word, G in seed_graphs.items():
    #    draw_graph(G, title=f"Rete semantica: {seed_word}",
    #              filename=f"rete_semantica_{seed_word}.png")