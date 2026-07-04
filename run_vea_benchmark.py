#!/usr/bin/env python3
"""
VetExamBench-60 — ICVA VEA Benchmark Runner
=============================================
Runs 60 official ICVA VEA sample questions against an AI model via the
Anthropic API. Produces a scored report with per-category breakdown,
confidence intervals, and pass/fail verdict.

Usage:
    # Full 60-question run (includes 7 image-dependent questions as text-only)
    python run_vea_benchmark.py --api-key sk-ant-... --model claude-sonnet-4-6

    # Text-only: skip image-dependent questions (53 questions)
    python run_vea_benchmark.py --api-key sk-ant-... --skip-image-questions

    # Dry-run: print questions without calling the API
    python run_vea_benchmark.py --dry-run

    # Use a specific JSON file
    python run_vea_benchmark.py --api-key sk-ant-... --questions path/to/vea_benchmark_60.json

Requirements:
    pip install anthropic  (or: pip install requests)

Source: ICVA VEA® Sample Questions — © ICVA. Used for non-commercial evaluation.
"""

import json
import os
import re
import sys
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Confidence interval (Wilson score)
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Load questions
# ---------------------------------------------------------------------------

def load_questions(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build prompt for a single question
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are taking a veterinary medicine multiple-choice exam. "
    "For each question, reply with ONLY the letter of the correct answer "
    "(A, B, C, D, E, or F). No explanation, no punctuation — just the letter."
)


def build_user_prompt(q: dict) -> str:
    lines = [f"Question {q['id']}: {q['question']}", ""]
    for letter, text in q["options"].items():
        lines.append(f"({letter}) {text}")
    if q.get("requires_image"):
        note = q.get("image_note", "")
        if note:
            lines.append(f"\n[Image context: {note}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Call the Anthropic API
# ---------------------------------------------------------------------------

def call_anthropic(api_key: str, model: str, question_prompt: str) -> str:
    """Call Anthropic Messages API and return the raw text response."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=8,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": question_prompt}],
        )
        return response.content[0].text.strip()
    except ImportError:
        pass

    # Fallback: raw requests
    import requests
    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 8,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": question_prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"].strip()


def extract_answer(raw: str) -> str:
    """Extract a single letter A-F from the model's response."""
    raw = raw.strip().upper()
    # If it's already a single letter
    if len(raw) == 1 and raw in "ABCDEF":
        return raw
    # Try to find (X) or just the first capital letter A-F
    match = re.search(r"\(?([A-F])\)?", raw)
    if match:
        return match.group(1)
    return raw[:1] if raw else "?"


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    questions_path: str,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
    skip_image: bool = False,
    dry_run: bool = False,
) -> dict:
    data = load_questions(questions_path)
    meta = data["metadata"]
    questions = data["questions"]

    if skip_image:
        questions = [q for q in questions if not q.get("requires_image")]

    total = len(questions)
    correct = 0
    wrong = []
    results_per_q = []
    category_stats: dict[str, dict] = {}

    print(f"\n{'='*66}")
    print(f"  VetExamBench-60  |  Model: {model}")
    print(f"  Questions: {total}  |  {'TEXT-ONLY' if skip_image else 'FULL (incl. image-context)'}")
    print(f"{'='*66}\n")

    for i, q in enumerate(questions, 1):
        prompt = build_user_prompt(q)
        cat = q.get("category", "Unknown")

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0, "wrong_ids": []}
        category_stats[cat]["total"] += 1

        if dry_run:
            print(f"  [{i:2d}/{total}] Q{q['id']:2d}  ({cat})  — DRY RUN")
            model_answer = "?"
            is_correct = False
        else:
            raw = call_anthropic(api_key, model, prompt)
            model_answer = extract_answer(raw)
            is_correct = model_answer == q["correct"]

            if is_correct:
                correct += 1
                category_stats[cat]["correct"] += 1
                mark = "✅"
            else:
                wrong.append({
                    "id": q["id"],
                    "category": cat,
                    "subcategory": q.get("subcategory", ""),
                    "expected": q["correct"],
                    "got": model_answer,
                    "question_snippet": q["question"][:80],
                    "requires_image": q.get("requires_image", False),
                })
                category_stats[cat]["wrong_ids"].append(q["id"])
                mark = "❌"

            print(f"  [{i:2d}/{total}] Q{q['id']:2d}  {mark}  model={model_answer}  key={q['correct']}  ({cat})")

        results_per_q.append({
            "id": q["id"],
            "model_answer": model_answer,
            "correct_answer": q["correct"],
            "is_correct": is_correct,
            "category": cat,
        })

    # Compute stats
    pct = (correct / total * 100) if total > 0 else 0
    ci_lo, ci_hi = wilson_ci(correct, total)
    passed = pct >= meta.get("pass_threshold_pct", 70)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "source": meta.get("title", "VEA Benchmark"),
        "total_questions": total,
        "image_questions_skipped": skip_image,
        "correct": correct,
        "wrong_count": total - correct,
        "score_pct": round(pct, 1),
        "ci_95_low": round(ci_lo * 100, 1),
        "ci_95_high": round(ci_hi * 100, 1),
        "pass_threshold_pct": meta.get("pass_threshold_pct", 70),
        "passed": passed,
        "category_breakdown": {},
        "wrong_answers": wrong,
        "results_per_question": results_per_q,
    }

    # Category breakdown
    for cat, stats in sorted(category_stats.items()):
        cat_pct = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
        report["category_breakdown"][cat] = {
            "total": stats["total"],
            "correct": stats["correct"],
            "pct": round(cat_pct, 1),
            "wrong_question_ids": stats["wrong_ids"],
        }

    # Print summary
    print(f"\n{'='*66}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*66}")
    print(f"  Score:     {correct}/{total}  ({pct:.1f}%)")
    print(f"  95% CI:    [{ci_lo*100:.1f}% — {ci_hi*100:.1f}%]")
    print(f"  Threshold: ≥{meta.get('pass_threshold_pct', 70)}%")
    print(f"  Verdict:   {'✅ PASSED' if passed else '❌ FAILED'}")
    print()
    print(f"  Category Breakdown:")
    print(f"  {'Category':<22} {'Score':>8}  {'Pct':>6}")
    print(f"  {'-'*22} {'-'*8}  {'-'*6}")
    for cat, stats in sorted(report["category_breakdown"].items()):
        s = f"{stats['correct']}/{stats['total']}"
        print(f"  {cat:<22} {s:>8}  {stats['pct']:>5.1f}%")

    if wrong:
        print(f"\n  Wrong Answers ({len(wrong)}):")
        for w in wrong:
            img_flag = " 📷" if w["requires_image"] else ""
            print(f"    Q{w['id']:2d} [{w['category']}/{w['subcategory']}]"
                  f"  expected={w['expected']} got={w['got']}{img_flag}")
    print(f"{'='*66}\n")

    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VetExamBench-60: ICVA VEA Benchmark Runner"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("ANTHROPIC_API_KEY"),
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model to test (default: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--questions",
        default=str(Path(__file__).parent / "vea_benchmark_60.json"),
        help="Path to the benchmark JSON file",
    )
    parser.add_argument(
        "--skip-image-questions",
        action="store_true",
        help="Skip the 7 image-dependent questions (run 53 text-only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print questions without calling the API",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the JSON report (default: benchmark_report_<timestamp>.json)",
    )

    args = parser.parse_args()

    if not args.dry_run and not args.api_key:
        print("Error: --api-key or ANTHROPIC_API_KEY required (unless --dry-run)")
        sys.exit(1)

    report = run_benchmark(
        questions_path=args.questions,
        api_key=args.api_key,
        model=args.model,
        skip_image=args.skip_image_questions,
        dry_run=args.dry_run,
    )

    # Save report
    if args.output:
        out_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"benchmark_report_{ts}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved → {out_path}")


if __name__ == "__main__":
    main()
