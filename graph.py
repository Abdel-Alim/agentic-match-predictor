"""LangGraph orchestration of the prediction agents.

Graph topology:

    scout -> analyst -> predictor -> critic --(fixable issue)--> predictor
                                       |
                                       +--> pundit -> publisher -> END

The critic/predictor edge is the agentic loop: outputs that fail validation
are sent back with directives before anything is published.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents import (
    analyst_agent,
    critic_agent,
    predictor_agent,
    publisher_agent,
    pundit_agent,
    route_after_critic,
    scout_agent,
)


class RunState(TypedDict, total=False):
    run_date: str
    no_cache: bool
    force_retrain: bool
    fixtures: list[dict]
    fixture_date: str
    scout_notes: list[str]
    degraded: bool
    analyzed: int
    model_metrics: dict
    retrained: bool
    predictions: list[dict]
    critiques: list[str]
    critic_directives: list[str]
    retries: int
    warnings: list[str]
    summary: list[str]
    outputs: list[str]
    report: list[str]


def build_graph():
    g = StateGraph(RunState)
    g.add_node("scout", scout_agent)
    g.add_node("analyst", analyst_agent)
    g.add_node("predictor", predictor_agent)
    g.add_node("critic", critic_agent)
    g.add_node("pundit", pundit_agent)
    g.add_node("publisher", publisher_agent)

    g.set_entry_point("scout")
    g.add_edge("scout", "analyst")
    g.add_edge("analyst", "predictor")
    g.add_edge("predictor", "critic")
    g.add_conditional_edges("critic", route_after_critic, {
        "predictor": "predictor",
        "pundit": "pundit",
    })
    g.add_edge("pundit", "publisher")
    g.add_edge("publisher", END)
    return g.compile()
