import pandas as pd
import openai
import faiss
import numpy as np
from tqdm import tqdm
from parameters import *

openai.api_key = OPENAI_API_KEY

def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

def get_embeddings_tqdm(text):
    embeddings = []
    print("Start converting the data into embedding vector...")
    for doc in tqdm(text, desc="Embedding "):
        embeddings.append(get_embedding(doc))
    return embeddings


if __name__ == '__main__':

    file = input("Enter csv file as database:\n")
    documents = doc_process(file)

    embeddings = get_embeddings_tqdm(documents)

    #turn the embedding vector into NumPy array
    embedding_dim = len(embeddings[0])  # 嵌入的向量維度
    index = faiss.IndexFlatL2(embedding_dim)  # 使用 L2 距離進行檢索

    #put hte data vector into model
    index.add(np.array(embeddings, dtype='float32'))

    query = input("Etr qtn abt csv file:\n")

    #turn the qtn into vector
    query_embedding = np.array([get_embedding(query)], dtype='float32')

    # 在 FAISS 中檢索最相似的文檔
    k = 2  # 檢索前2個最相關的文檔
    distances, indices = index.search(query_embedding, k)

    #get the data
    relevant_docs = [documents[i] for i in indices[0]]
    print("The data we found:")
    print(*relevant_docs, sep="\n")
    print()

    context = " ".join(relevant_docs)

    #output
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Answer the question based on the following data: {context}\n\nquestion: {query}"}
        ],
        max_tokens=150
    )

    print("Response", encoder(response.choices[0].message["content"]))
