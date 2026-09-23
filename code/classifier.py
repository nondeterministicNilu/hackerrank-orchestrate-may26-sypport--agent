import ollama
import json

MODEL_NAME = "llama3.1:8b"


def classify_ticket(ticket, retrieved_documents):

    issue = str(ticket["Issue"])
    subject = str(ticket["Subject"])
    company = str(ticket["Company"])

    # Get top 5 relevant documents
    context = "\n\n".join(
        retrieved_documents["content"].tolist()
    )

    # print(context)

    prompt = f"""
Classify the support ticket.

SUBJECT:
{subject}

ISSUE:
{issue}

Determine exactly one request_type.

Rules:

1. bug
Choose "bug" when an existing feature or functionality is
broken, failing, throwing an error, crashing, or behaving
unexpectedly.

2. feature_request
Choose "feature_request" when the customer wants a NEW
feature, capability, integration, option, or enhancement.

3. product_issue
Choose "product_issue" when the customer is asking how to use,
configure, or understand an EXISTING feature.

4. invalid
Choose "invalid" when the request is unrelated to the
supported product or is not a valid support request.

IMPORTANT:
- Focus ONLY on the customer's SUBJECT and ISSUE.
- Do NOT assume something is a product_issue just because
  the topic relates to an existing product feature.
- If something is broken or produces an error, classify it as bug.
- If the customer asks for something new, classify it as feature_request.
- If the customer asks how to use an existing feature,
  classify it as product_issue.

Examples:

Subject: Password reset problem
Issue: I get an error every time I try to reset my password
→ bug

Subject: Add Google login
Issue: Can you add Google authentication?
→ feature_request

Subject: How do I reset my password?
Issue: Please tell me how to change my password
→ product_issue

Subject: Python help
Issue: Can you write a Python program to sort a list?
→ invalid

Return ONLY valid JSON:

{{
    "request_type": "bug"
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

    return json.loads(response["message"]["content"])


def classify_product_area(ticket, retrieved_documents):

    subject = str(ticket["Subject"])
    issue = str(ticket["Issue"])

    context = "\n\n".join(
        retrieved_documents["content"].head(5).tolist()
    )

    prompt = f"""
Determine the most relevant product area for this support ticket.

SUBJECT:
{subject}

ISSUE:
{issue}

RELEVANT SUPPORT DOCUMENTATION:
{context}

Choose the product area based only on the ticket and
provided documentation.

*** This should be the most relevant support category / domain area
For exampple,
screen, privacy, travel_Support, general_support, conversation_management, community etc

Do not use outside knowledge.

Return ONLY valid JSON:

{{
    "product_area": "..."
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

    return json.loads(response["message"]["content"])


def generate_response(ticket, retrieved_documents):

    subject = str(ticket["Subject"])
    issue = str(ticket["Issue"])

    context = "\n\n".join(
        retrieved_documents["content"].head(5).tolist()
    )

    prompt = f"""
Answer the customer's support request.

SUBJECT:
{subject}

ISSUE:
{issue}

SUPPORT DOCUMENTATION:
{context}

Rules:

- Use ONLY the provided support documentation.
- Do not use outside knowledge.
- Do not invent policies, features, instructions, or solutions.
- Give a concise and helpful user-facing response.
- If the documentation does not contain enough information
  to answer the request, say that the available documentation
  does not provide enough information.

Return ONLY valid JSON:

{{
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
            "temperature": 2.0
        }
    )

    return json.loads(response["message"]["content"])

def validate_status(ticket, retrieved_documents, request_type):

    if request_type == "invalid":
        return {
            "status": "escalated",
            "justification": "The request is outside the supported support scope."
        }

    if retrieved_documents.empty:
        return {
            "status": "escalated",
            "justification": "No relevant support documentation was found."
        }

    if retrieved_documents["score"].max() == 0:
        return {
            "status": "escalated",
            "justification": "No sufficiently relevant support documentation was found."
        }

    return {
        "status": "replied",
        "justification": "Relevant support documentation was found."
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

    # Classify request type
    # request_type = classify_ticket(ticket, retrieved_documents)

    # print("\nRequest Type:")
    # print(request_type)

    # # Classify product area
    # product_area = classify_product_area(ticket, retrieved_documents)

    # print("\nProduct Area:")
    # print(product_area)

    # # Validate status
    # status = validate_status(
    #     ticket,
    #     retrieved_documents,
    #     request_type["request_type"]
    # )

    # print("\nStatus:")
    # print(status)

    # Generate response
    response = generate_response(ticket, retrieved_documents)
        
    print("\nResponse:")
    print(response)