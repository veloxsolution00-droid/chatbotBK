"""Centralized prompt instructions for grounded multi-part answers."""
from app.rag.planner import task_guidance
from app.reasoning.scenario_analyzer import scenario_instructions
from app.reasoning.comparison import comparison_instructions
from app.reasoning.summarizer import summarization_instructions


def answer_instructions(intent: str) -> str:
    extras = []
    if intent == "scenario": extras.append(scenario_instructions())
    if intent == "comparison": extras.append(comparison_instructions())
    if intent == "summary": extras.append(summarization_instructions())
    return " ".join([task_guidance(intent), *extras])
