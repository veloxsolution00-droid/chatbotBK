"""Run a small JSON-based regression suite against the local chat API.

Usage: python evaluation/run_api_evaluation.py evaluation/my_cases.json
The user must upload/index the test documents before running the suite.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import requests

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--report", type=Path, default=Path("evaluation/latest_report.json"))
    args = parser.parse_args()
    try:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
    except Exception as exc:
        parser.error(f"Cannot read evaluation cases: {exc}")
    if not isinstance(cases, list) or not cases:
        parser.error("Evaluation file must be a non-empty JSON array.")
    rows = []
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or not isinstance(case.get("question"), str):
            rows.append({"case": index, "passed": False, "error": "Each case requires a question string."})
            continue
        started = time.perf_counter()
        try:
            response = requests.post(args.api.rstrip("/") + "/chat", json={
                "question": case["question"], "history": case.get("history", [])}, timeout=args.timeout)
            response.raise_for_status()
            payload = response.json()
            answer = str(payload.get("answer", ""))
            sources = payload.get("sources", []) or []
            checks = {}
            if "must_abstain" in case:
                checks["abstention"] = bool(payload.get("abstained")) is bool(case["must_abstain"])
            if case.get("must_contain"):
                checks["must_contain"] = all(str(term).lower() in answer.lower() for term in case["must_contain"])
            if case.get("must_not_contain"):
                checks["must_not_contain"] = all(str(term).lower() not in answer.lower() for term in case["must_not_contain"])
            if case.get("expected_sources"):
                actual = {str(src.get("filename", "")).lower() for src in sources}
                checks["expected_sources"] = all(str(name).lower() in actual for name in case["expected_sources"])
            if case.get("expected_question_type"):
                checks["question_type"] = payload.get("question_type") == case["expected_question_type"]
            passed = bool(checks) and all(checks.values()) and payload.get("status") != "processing_error"
            rows.append({"case": index, "name": case.get("name", f"case-{index}"), "passed": passed,
                         "checks": checks, "status": payload.get("status"), "abstained": payload.get("abstained"),
                         "question_type": payload.get("question_type"), "response_seconds": payload.get("response_seconds", round(time.perf_counter()-started, 3)),
                         "source_filenames": [s.get("filename") for s in sources], "answer": answer})
        except Exception as exc:
            rows.append({"case": index, "name": case.get("name", f"case-{index}"), "passed": False, "error": str(exc)})
        print(f"{'PASS' if rows[-1].get('passed') else 'FAIL'} case {index}: {rows[-1].get('name', case.get('name', ''))}")
    scored = [r for r in rows if isinstance(r.get("passed"), bool)]
    report = {"case_count": len(rows), "passed": sum(bool(r.get("passed")) for r in scored),
              "failed": sum(not bool(r.get("passed")) for r in scored),
              "pass_rate": (sum(bool(r.get("passed")) for r in scored) / len(scored)) if scored else None,
              "results": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nPass rate: {report['passed']}/{report['case_count']} ({(report['pass_rate'] or 0):.1%})")
    print(f"Report: {args.report}")
    return 0 if report["failed"] == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
