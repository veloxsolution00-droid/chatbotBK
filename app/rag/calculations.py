"""Small safe arithmetic evaluator; never uses eval or executes arbitrary code."""
from __future__ import annotations
import ast
import operator
import re
import json

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

def evaluate_expression(expression: str) -> float:
    """Evaluate a numeric arithmetic expression with strict AST and size limits."""
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 160:
        raise ValueError("Expression is empty or too long")
    if not re.fullmatch(r"[\d\s.+*/()%\-]+", expression):
        raise ValueError("Only numeric arithmetic operators are allowed")
    node = ast.parse(expression, mode="eval")
    def visit(n):
        if isinstance(n, ast.Expression): return visit(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            if abs(n.value) > 1e12: raise ValueError("Numeric input exceeds the supported range")
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            left, right = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Pow) and abs(right) > 8: raise ValueError("Exponent is too large")
            value = _BINOPS[type(n.op)](left, right)
            if abs(value) > 1e15: raise ValueError("Result exceeds the supported range")
            return value
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UNARY: return _UNARY[type(n.op)](visit(n.operand))
        raise ValueError("Unsupported arithmetic expression")
    try:
        result = visit(node)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero") from exc
    return result

def direct_calculation(question: str) -> dict | None:
    """Calculate only explicit expressions or simple 'X percent of Y' questions."""
    q = (question or "").strip()
    pct = re.search(r"(?i)(-?\d+(?:\.\d+)?)\s*%\s+of\s+(-?\d+(?:\.\d+)?)", q)
    if pct:
        rate, base = map(float, pct.groups())
        return {"expression": f"({rate} / 100) * {base}", "result": (rate / 100) * base, "kind": "percentage"}
    # Require a calculation cue and an explicit arithmetic expression, preventing
    # accidental evaluation of document identifiers or dates.
    if not re.search(r"(?i)\b(calculate|compute|what is|evaluate)\b", q):
        return None
    match = re.search(r"(?<![\w.])\d+(?:\.\d+)?(?:\s*[+*/%()-]\s*\d+(?:\.\d+)?)+(?![\w.])", q)
    if not match:
        return None
    expr = match.group(0)
    try:
        result = evaluate_expression(expr)
    except (ValueError, SyntaxError):
        return None
    return {"expression": expr, "result": result, "kind": "arithmetic"}


def apply_evidence_calculation_plan(plan_text: str, evidence: str, question: str, source_count: int) -> dict | None:
    """Validate and evaluate a model-proposed numeric formula against supplied evidence.

    The model may select a formula, but it cannot introduce numeric operands that
    do not appear in the question/evidence (except the conventional 100 in a
    percentage expression). Formula selection remains a semantic risk.
    """
    if not isinstance(plan_text, str): return None
    raw = plan_text.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
    try:
        plan = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(plan, dict) or not isinstance(plan.get("expression"), str): return None
    expression = plan["expression"].strip()
    refs = plan.get("source_refs", [])
    if not isinstance(refs, list) or any(not isinstance(x, int) or x < 1 or x > source_count for x in refs):
        return None
    if not refs:
        return None
    context = f"{question}\n{evidence}"
    context_numbers = {float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", context)}
    expression_numbers = [float(x) for x in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", expression)]
    allow_100 = "%" in question or "percent" in question.lower() or "percentage" in question.lower()
    if any(number not in context_numbers and not (allow_100 and number == 100.0) for number in expression_numbers):
        return None
    try:
        value = evaluate_expression(expression)
    except (ValueError, SyntaxError):
        return None
    return {"expression": expression, "result": value, "source_refs": sorted(set(refs)),
            "kind": "document_evidence", "unit": str(plan.get("unit", "")).strip()[:40]}
