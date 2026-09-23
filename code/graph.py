from typing import TypedDict

from langgraph.graph import StateGraph, END

from retriever import search_documents
from classifier import (
    classify_ticket,
    validate_status
)


class TicketState(TypedDict):
    ticket: dict
    retrieved_documents: object
    request_type: str
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
    state["product_area"] = result["product_area"]
    state["response"] = result["response"]

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


def build_graph():

    graph = StateGraph(TicketState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("classify", classify)
    graph.add_node("validate", validate)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "classify")
    graph.add_edge("classify", "validate")
    graph.add_edge("validate", END)

    return graph.compile()