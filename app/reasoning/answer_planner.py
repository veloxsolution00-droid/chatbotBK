"""Plan completion requirements for multi-part questions."""

def build_answer_plan(subquestions: list[str]) -> dict:
    requirements = [{"id": i + 1, "question": q, "required": True} for i, q in enumerate(subquestions)]
    return {"requirements": requirements, "total_requirements": len(requirements),
            "completion_rule": "Do not mark a requirement answered unless evidence was retrieved and its answer is supported."}


def coverage_summary(results: list[dict]) -> dict:
    total = len(results)
    answered = sum(1 for item in results if item.get("status") == "success" and not item.get("abstained", False))
    return {"total_requirements": total, "answered_requirements": answered,
            "unanswered_requirements": total - answered,
            "coverage_ratio": round(answered / total, 4) if total else 1.0,
            "per_requirement": [{"question": r.get("question"), "status": r.get("status"),
                                 "evidence_chunks": len(r.get("sources", [])),
                                 "answered": r.get("status") == "success" and not r.get("abstained", False)} for r in results]}
