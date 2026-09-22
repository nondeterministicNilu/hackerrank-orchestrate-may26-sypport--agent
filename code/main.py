import pandas as pd
from pathlib import Path
from retriever import load_documents
from agent import process_ticket
from graph import build_graph

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "support_tickets" / "support_tickets.csv"
OUTPUT_FILE = PROJECT_ROOT / "support_tickets" / "output.csv"


# Load tickets
tickets = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(tickets)} tickets")

# Load support documents
documents = load_documents()
print(f"Loaded {len(documents)} support documents")

# Build LangGraph
graph = build_graph()


def process_ticket(ticket):

    state = {
        "ticket": ticket.to_dict(),
        "documents": documents
    }

    result = graph.invoke(state)

    return pd.Series({
        "Response": result.get("response", ""),
        "Product Area": result.get("product_area", ""),
        "Status": result.get("status", ""),
        "Request Type": result.get("request_type", ""),
        "Justification": result.get("justification", "")
    })


# Process all tickets
tickets[
    [
        "Response",
        "Product Area",
        "Status",
        "Request Type",
        "Justification"
    ]
] = tickets.apply(
    process_ticket,
    axis=1
)


# Save output
tickets.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Output saved to: {OUTPUT_FILE}")
