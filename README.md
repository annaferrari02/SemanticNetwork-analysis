# Semantic Dynamics in LLMs

This repository contains the code and report for the research project "Semantic Dynamics in LLMs: A Comparison Between Spreading Activation and Vector-Based Similarity" by Anna Ferrari, University of Trento.

## Project Overview

This research investigates how Large Language Models (LLMs) organize and process semantic information by comparing three approaches to semantic similarity:

1. **Spreading activation** in LLM-generated semantic networks
2. **Human-like judgments** of word pair associations (simulated by LLMs)
3. **Vector-based similarity** measurements using word embeddings

The study explores whether LLMs rely more on static geometric proximity in embedding space or on dynamic associative processes that are characteristic of human cognition.

## Repository Contents

- **Report**: Complete research findings and methodology
  - `Report_final.pdf`
  - 'ReportCS.ipynb' 

- **Code Files**:
  - `generate_network.py`: Generates semantic networks using Mistral AI
  - `generate_likertscores.py`: Produces similarity judgments on a Likert scale
  - `spreading_activation.py`: Simulates spreading activation processes in the network
  - `cosine_sim_w2v.py`: Calculates cosine similarity using Word2Vec embeddings

## Key Findings

- Cosine similarity correlates more strongly with LLM-generated similarity judgments (r=0.467) than spreading activation values (r=0.37)
- This suggests that LLMs rely more on static geometric proximity in embedding space than on dynamic associative processes characteristic of human cognition
- While embedding models are trained on co-occurrence data, they transform this information into distributed semantic representations that capture second-order relationships between concepts

## Requirements

See `requirements.txt` for all necessary dependencies.

## Usage

1. Generate the semantic network:
```
python generate_network.py
```

2. Generate Likert scores for word pairs:
```
python generate_likertscores.py
```

3. Run spreading activation simulation:
```
python spreading_activation.py
```

4. Calculate word embedding similarities:
```
python cosine_sim_w2v.py
```
