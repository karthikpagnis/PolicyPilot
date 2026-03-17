from langgraph.graph import StateGraph, END
from typing import TypedDict
from agents.segregator import segregator_agent
from agents.id_agent import id_agent
from agents.discharge_agent import discharge_agent
from agents.bill_agent import bill_agent
from agents.bank_agent import bank_agent


class ClaimState(TypedDict):
    """Shared state that flows through the LangGraph workflow."""
    claim_id: str
    pdf_bytes: bytes
    page_images: dict
    page_classifications: dict
    routing: dict
    id_data: dict
    discharge_data: dict
    bill_data: dict
    bank_data: dict
    final_result: dict


def aggregator(state: ClaimState) -> ClaimState:
    """
    Combines outputs from all extraction agents into final JSON response.
    """
    print("Aggregator: Combining results")

    final = {
        "claim_id": state["claim_id"],
        "total_pages_processed": len(state.get("page_classifications", {})),
        "page_classifications": state["page_classifications"],
        "extracted_data": {
            "identity": state.get("id_data", {}),
            "discharge_summary": state.get("discharge_data", {}),
            "itemized_bill": state.get("bill_data", {}),
            "cheque_or_bank_details": state.get("bank_data", {}),
        }
    }

    print("Aggregation complete")
    return {**state, "final_result": final}


def build_workflow():
    """
    Builds the LangGraph state machine.

    Flow: Segregator -> [ID, Discharge, Bill, Bank] (parallel) -> Aggregator -> END
    """
    graph = StateGraph(ClaimState)

    graph.add_node("segregator", segregator_agent)
    graph.add_node("id_agent", id_agent)
    graph.add_node("discharge", discharge_agent)
    graph.add_node("bill", bill_agent)
    graph.add_node("bank", bank_agent)
    graph.add_node("aggregator", aggregator)

    graph.set_entry_point("segregator")

    # Fan out to all 4 agents in parallel
    graph.add_edge("segregator", "id_agent")
    graph.add_edge("segregator", "discharge")
    graph.add_edge("segregator", "bill")
    graph.add_edge("segregator", "bank")

    # All agents converge into the aggregator
    graph.add_edge("id_agent", "aggregator")
    graph.add_edge("discharge", "aggregator")
    graph.add_edge("bill", "aggregator")
    graph.add_edge("bank", "aggregator")

    graph.add_edge("aggregator", END)

    return graph.compile()


claim_workflow = build_workflow()
