import csv
# Carica il modello Word2Vec
w2v_model = api.load('glove-twitter-50')


def calculate_similarity(word1, word2):
    try:
        return w2v_model.similarity(word1, word2)
    except KeyError:
        return None  # Restituisce None se una delle parole non è nel vocabolario


def add_similarity_csv(input_file):

    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        rows = list(reader)


    for i, row in enumerate(rows):
        if i == 0:

            row.append('Similarity')
        else:
            word1 = row[0]
            word2 = row[1]
            similarity = calculate_similarity(word1, word2)
            row.append(similarity)


    with open(input_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(rows)

input_file = '/content/prova_likert300.csv'  # Sostituisci con il percorso del tuo file di input
add_similarity_csv(input_file)
import pandas as pd
df = pd.read_csv('/content/prova_likert300.csv')
print(df.head())
df2= pd.read_csv("/content/spreading_results.csv")
print(df2.head())

# Extract the desired columns from df
df_subset = df[['LikertScore', 'Similarity']]

# Append the subset to df2
df2 = pd.concat([df2, df_subset], axis=1)

print(df2.head())


total= df2
total.to_csv('output.csv', index=False)  # Save to 'output.csv' without the index