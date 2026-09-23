import os
from pathlib import Path
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import psycopg2
from pgvector.psycopg2 import register_vector
import ollama
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

embedding_model_name = os.getenv("OLLAMA_EMBEDDING_MODEL")

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

OPTIMAL_SCORE = 0.1  # Adjust this threshold based on your requirements

# Load stopwords
stop_words = set(stopwords.words("english"))

def tokenize(text): 
    tokens = word_tokenize(text.lower())
    return [ token for token in tokens if token.isalnum() and token not in stop_words ]


def load_documents():

    documents = []

    # Go through each company's folder
    for company_folder in DATA_FOLDER.iterdir():

        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
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


                except Exception as e:
                    print(f"Could not read {file}: {e}")
        conn.commit()
        cursor.close()
        conn.close()
    return pd.DataFrame(documents)

def search_documents(query, company=None, top_k=5, score_threshold=OPTIMAL_SCORE):

    # Generate query embedding using Ollama
    embedding_response = ollama.embed(
        model=embedding_model_name,
        input=query
    )

    query_embedding = embedding_response["embeddings"][0]

    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    sql = """
        WITH keyword_results AS (
            SELECT
                company,
                file_name,
                content,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank_cd(
                        to_tsvector('english', content),
                        plainto_tsquery('english', %s)
                    ) DESC
                ) AS keyword_rank
            FROM "dbo"."data"
            WHERE
                LOWER(company) = LOWER(%s)
                AND to_tsvector('english', content)
                    @@ plainto_tsquery('english', %s)
            LIMIT 50
        ),

        vector_results AS (
            SELECT
                company,
                file_name,
                content,
                ROW_NUMBER() OVER (
                    ORDER BY content_vector <=> %s::vector
                ) AS vector_rank
            FROM "dbo"."data"
            WHERE LOWER(company) = LOWER(%s)
            LIMIT 50
        )

        SELECT
            COALESCE(k.company, v.company) AS company,
            COALESCE(k.file_name, v.file_name) AS file_name,
            COALESCE(k.content, v.content) AS content,

            COALESCE(1.0 / (60 + k.keyword_rank), 0) +
            COALESCE(1.0 / (60 + v.vector_rank), 0) AS score

        FROM keyword_results k

        FULL OUTER JOIN vector_results v
            ON k.company = v.company
            AND k.file_name = v.file_name

        ORDER BY score DESC
        LIMIT %s
    """

    params = (
        query,                 
        company, 
        query, 

        str(query_embedding), 
        company, 

        top_k
    )


    embedding_response = ollama.embed(
        model=embedding_model_name,
        input=query
    )

    query_embedding = embedding_response["embeddings"][0]

    results = pd.read_sql_query(
        sql,
        conn,
        params=params
    )

    conn.close()

    return results



if __name__ == "__main__":

    results = search_documents(
        "tell me about hackerrank exam platform",
        company="HackerRank"
    )
    print("Retrieved Documents:", results)
