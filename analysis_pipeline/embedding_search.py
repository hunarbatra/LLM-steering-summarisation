import openai 
from openai.embeddings_utils import get_embedding, cosine_similarity

import pandas as pd

import time

embedding_model = "text-embedding-ada-002"

def generate_emb(text):
    time.sleep(2)
    return get_embedding(text, engine=embedding_model)

def semantic_search(df, c):
    df["similarity"] = [cosine_similarity(x, generate_emb(c)) for x in df['embeddings']]
    results = (
        df.sort_values("similarity", ascending=False)
    )
    return results.iloc[0]['summary'], results.iloc[0]['similarity']

def embedding_search_runner(summary, claims_list, api_key):
    openai.api_key = api_key
    embedding_val_res = []
    if summary[-1] == '.':
        summary = summary[:-1]
    summary_embedding = [generate_emb(chunk.strip()) for chunk in summary.split('.') if chunk != '']
    df = pd.DataFrame({'summary': summary.split('.'), 'embeddings': summary_embedding})
    for c in claims_list:
        similar_sentence, similarity_score = semantic_search(df, c)
        print('Most similar sentence to the claim: ' + c + 'is: ' + similar_sentence)
        print('Similarity Score: ' + str(similarity_score))
        embedding_val_res.append(1) if similarity_score >= 0.85 else embedding_val_res.append(0)
    print(f'Embedding Search Validation Score: {sum(embedding_val_res)/len(embedding_val_res)}, {embedding_val_res}')
    return sum(embedding_val_res) / len(embedding_val_res)
