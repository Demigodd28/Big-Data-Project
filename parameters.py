import os
import pandas as pd

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

def doc_process(csv_file):
    df = pd.read_csv(csv_file)
    documents = df.apply(lambda row: ', '.join(row.values.astype(str)), axis=1)
    documents = documents.tolist()
    return documents

def encoder(text):
    return text.encode().decode()

if __name__ == '__main__':
    file = input()
    
    print(doc_process(file))
