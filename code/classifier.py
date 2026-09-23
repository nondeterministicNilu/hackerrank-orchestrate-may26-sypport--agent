import ollama
import json
import time
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = os.getenv("OLLAMA_MODEL")


def classify_ticket(ticket, retrieved_documents):

    start_time = time.time()

    issue = str(ticket["Issue"])
    subject = str(ticket["Subject"])
    company = str(ticket["Company"])

    context = "\n\n".join(
        retrieved_documents["content"].head(3).apply(
            lambda x: str(x)[:500]
        )
    )
    context = context + "|" + subject + "|" + issue

    prompt = f"""Use the following context and classify the tickets foe company {company}.
{context} 
REQUEST TYPE:
- bug = existing functionality is broken or failing
- feature_request = customer wants a new capability
- product_issue = customer asks how to use/configure an existing feature
- invalid = unrelated or unsupported request

PRODUCT AREA:
Choose the most relevant support category based on the ticket and documentation.

RESPONSE:
Answer using ONLY the documentation.
Do not invent information.
If the documentation is insufficient, say so.

Return ONLY JSON:

{{
    "request_type": "bug",
    "product_area": "general_support",
    "response": "..."
}}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0
        }
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Classification completed for {ticket['Subject']} in {elapsed_time:.2f} seconds.")

    print("Prompt tokens:", response.get("prompt_eval_count"))
    print("Output tokens:", response.get("eval_count"))
    print("Prompt eval time:", response.get("prompt_eval_duration"))
    print("Generation time:", response.get("eval_duration"))

    return json.loads(
        response["message"]["content"]
    )


def validate_status(ticket, retrieved_documents, request_type):

    if request_type == "invalid":
        return {
            "status": "escalated",
            "justification": (
                "The request is outside the supported support scope."
            )
        }

    if retrieved_documents.empty:
        return {
            "status": "escalated",
            "justification": (
                "No relevant support documentation was found."
            )
        }

    if retrieved_documents["score"].max() == 0:
        return {
            "status": "escalated",
            "justification": (
                "No sufficiently relevant support documentation "
                "was found."
            )
        }

    return {
        "status": "replied",
        "justification": (
            "Relevant support documentation was found."
        )
    }


if __name__ == "__main__":

    from retriever import search_documents

    ticket = {
        "Subject": "Password reset problem",
        "Issue": "I cannot reset my password",
        "Company": "HackerRank"
    }

    # Retrieve relevant documents
    query = f"{ticket['Subject']} {ticket['Issue']}"

    retrieved_documents = search_documents(
        query,
        company=ticket["Company"],
        top_k=5
    )

    print("\nRetrieved Documents:")

    print(
        retrieved_documents[
            ["company", "file_name", "score"]
        ]
    )

    # Single Ollama call
    result = classify_ticket(
        ticket,
        retrieved_documents
    )

    print("\nRequest Type:")
    print(result["request_type"])

    print("\nProduct Area:")
    print(result["product_area"])

    print("\nResponse:")
    print(result["response"])

    # Validate status
    status = validate_status(
        ticket,
        retrieved_documents,
        result["request_type"]
    )

    print("\nStatus:")
    print(status["status"])

    print("\nJustification:")
    print(status["justification"])