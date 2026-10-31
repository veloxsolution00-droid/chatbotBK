"""Compatibility export for the existing safe AST-based calculator."""
from app.rag.calculations import evaluate_expression, direct_calculation, apply_evidence_calculation_plan

__all__ = ["evaluate_expression", "direct_calculation", "apply_evidence_calculation_plan"]
