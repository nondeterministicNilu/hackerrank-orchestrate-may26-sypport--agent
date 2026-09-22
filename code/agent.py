from retriever import search_documents


def process_ticket(ticket, documents):

    issue = ticket["Issue"]
    subject = ticket["Subject"]
    company = ticket["Company"]

    # Combine subject and issue
    query = f"{subject} {issue}"

    # Retrieve relevant documents
    results = search_documents(
        query,
        documents,
        company=company,
        top_k=5
    )

    # Temporary values
    status = "escalated"
    product_area = ""
    response = ""
    justification = ""
    request_type = "invalid"

    # If no relevant documents are found
    if results.empty or results["score"].max() == 0:

        response = (
            "I could not find sufficient information in the available support documentation to answer this request."
        )

        justification = (
            "No relevant support documentation was found."
        )

        return {
            "status": status,
            "product_area": product_area,
            "response": response,
            "justification": justification,
            "request_type": request_type
        }

    # We have relevant documents
    best_document = results.iloc[0]

    product_area = best_document["file_name"]

    response = (
        "Relevant support documentation was found. "
        "A detailed response will be generated from the retrieved content."
    )

    justification = (
        f"Relevant documentation was found in "
        f"{best_document['file_name']}."
    )

    return {
        "status": status,
        "product_area": product_area,
        "response": response,
        "justification": justification,
        "request_type": request_type
    }