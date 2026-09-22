from pathlib import Path
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama

embedding_model_name = 'nomic-embed-text-v2-moe'

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

OPTIMAL_SCORE = 10

# Load stopwords
stop_words = set(stopwords.words("english"))

# conn = psycopg2.connect(
#     dbname="hackerrank-support-agent",
#     user="postgres",
#     password="Password@123",
#     host="localhost",
#     port="5432"
# )

# # Crucial: This ensures python understands the pgvector format
# register_vector(conn)
# cursor = conn.cursor()

def tokenize(text): 
    tokens = word_tokenize(text.lower())
    return [ token for token in tokens if token.isalnum() and token not in stop_words ]


def load_documents():

    documents = []

    # Go through each company's folder
    for company_folder in DATA_FOLDER.iterdir():

        conn = psycopg2.connect(
            dbname="hackerrank-support-agent",
            user="postgres",
            password="Password@123",
            host="localhost",
            port="5432"
        )

        # Crucial: This ensures python understands the pgvector format
        register_vector(conn)
        cursor = conn.cursor()

        company = company_folder.name
        print(f"Loading documents for company: {company}")

        # Read all files
        counter = 0
        total = sum(1 for _ in company_folder.rglob("*") if _.is_file())
        for file in company_folder.rglob("*"):

            if file.is_file():
                try:
                    counter += 1

                    content = file.read_text(encoding="utf-8", errors="ignore")

                    query = "INSERT INTO dbo.data (content_id, company, file_name, content, content_vector) VALUES (%s, %s, %s, %s, %s);"
                    id = f"{company}_{counter:03d}_{file.name.split('.')[0]}"

                    print(f"\rFile progress: {counter}/{total}\t{id}", end="", flush=True)

                    if len(content) > 0:  

                        response = ollama.embed(
                            model=embedding_model_name,
                            input=content
                        ) 

                        cursor.execute(query, (id.lower(), company, file.name, content, response['embeddings'][0]))
                    else:
                        cursor.execute(query, (id.lower(), company, file.name, content, np.zeros(768)))  # Insert an empty vector for empty content


                    # documents.append(
                    #     {
                    #         "company": company,
                    #         "file_name": file.name,
                    #         "content": content
                    #     }
                    # )

                except Exception as e:
                    print(f"Could not read {file}: {e}")
        conn.commit()
        cursor.close()
        conn.close()
    return pd.DataFrame(documents)


def search_documents(query, documents, company=None, top_k=5):

    # Filter by company if company is available
    if company and company != "None":
        results = documents[
            documents["company"].str.lower() == company.lower()
        ].copy()
    else:
        results = documents.copy()


    # Tokenize documents
    corpus = results["content"].apply(tokenize).tolist()

    # Create BM25 index
    bm25 = BM25Okapi(corpus)

    # Tokenize query
    query_tokens = tokenize(query)

    # Calculate BM25 scores
    scores = bm25.get_scores(query_tokens)

    results["score"] = scores

    # Sort by score
    results = results.sort_values(
        "score",
        ascending=False
    )

    results = results[results["score"] >= OPTIMAL_SCORE]
    
    # Return top results
    return results.head(top_k)




if __name__ == "__main__":

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    # # print(documents.head())

    # results = search_documents(
    #     "some random thought",
    #     documents,
    #     company="HackerRank"
    # )
    # print(results[["company", "file_name", "score"]])

    # print("Testing database connection...")
    # cursor.execute("SELECT * FROM dbo.data;")
    # print(cursor.fetchall())