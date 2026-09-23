from typing import TypedDict

from langgraph.graph import StateGraph, END

from retriever import search_documents
from classifier import (
    classify_ticket,
    classify_product_area,
    validate_status,
    generate_response
)


class TicketState(TypedDict):
    ticket: dict
    retrieved_documents: object
    request_type: dict
    status: str
    product_area: str
    response: str
    justification: str

    
def retrieve(state):

    ticket = state["ticket"]
    query = f"{ticket['Subject']} {ticket['Issue']}"
    retrieved_documents = search_documents(
        query,
        company=ticket["Company"],
        top_k=5
    )
    state["retrieved_documents"] = retrieved_documents
    return state


def classify(state):

    result = classify_ticket(
        state["ticket"],
        state["retrieved_documents"]
    )
    state["request_type"] = result["request_type"]
    return state

def product_area(state):

    result = classify_product_area(
        state["ticket"],
        state["retrieved_documents"]
    )
    state["product_area"] = result["product_area"]
    return state

def validate(state):

    result = validate_status(
        state["ticket"],
        state["retrieved_documents"],
        state["request_type"]
    )
    state["status"] = result["status"]
    state["justification"] = result["justification"]
    return state

def response(state):

    result = generate_response(
        state["ticket"],
        state["retrieved_documents"]
    )
    state["response"] = result["response"]
    return state

def build_graph():

    graph = StateGraph(TicketState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("classify", classify)
    graph.add_node("product_area", product_area)
    graph.add_node("validate", validate)
    graph.add_node("response", response)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "classify")
    graph.add_edge("classify", "product_area")
    graph.add_edge("product_area", "validate")
    graph.add_edge("validate", "response")
    graph.add_edge("response", END)

    return graph.compile()