from pathlib import Path
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_ROOT / "data"

OPTIMAL_SCORE = 10

# Load stopwords
stop_words = set(stopwords.words("english"))

def tokenize(text): 
    tokens = word_tokenize(text.lower())
    return [ token for token in tokens if token.isalnum() and token not in stop_words ]


def load_documents():

    documents = []

    # Go through each company's folder
    for company_folder in DATA_FOLDER.iterdir():

        company = company_folder.name

        # Read all files
        for file in company_folder.rglob("*"):

            if file.is_file():
                try:
                    content = file.read_text(encoding="utf-8", errors="ignore")
                    
                    documents.append(
                        {
                            "company": company,
                            "file_name": file.name,
                            "content": content
                        }
                    )

                except Exception as e:
                    print(f"Could not read {file}: {e}")

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
    # print(documents.head())

    results = search_documents(
        "some random thought",
        documents,
        company="HackerRank"
    )
    print(results[["company", "file_name", "score"]])