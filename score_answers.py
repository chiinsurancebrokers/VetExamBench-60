#!/usr/bin/env python3
"""
VEA Benchmark Offline Scorer
=============================
Scores a set of answers against the official ICVA answer key without
needing an API call. Useful for grading manually collected model outputs.

Usage:
    # Score from a text file (one letter per line, 60 lines):
    python score_answers.py --answers my_answers.txt

    # Score from command-line:
    python score_answers.py --answers-inline "D,B,A,C,D,B,C,A,A,C,A,B,D,A,D,A,B,B,A,A,C,E,A,B,C,B,D,C,D,A,A,B,B,B,A,A,A,D,D,B,C,D,D,D,D,A,A,A,C,C,D,A,D,E,C,D,A,A,B,A"

    # Score and show only wrong answers:
    python score_answers.py --answers my_answers.txt --errors-only
"""

import json
import math
import sys
import argparse
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def load_answer_key(json_path: str) -> list[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["questions"]


def load_answers_from_file(path: str) -> list[str]:
    """Load answers from a text file. Accepts one letter per line or comma-separated."""
    with open(path, "r") as f:
        content = f.read().strip()
    if "," in content:
        return [a.strip().upper() for a in content.split(",")]
    return [line.strip().upper() for line in content.splitlines() if line.strip()]


def score(questions: list[dict], answers: list[str], errors_only: bool = False):
    if len(answers) != len(questions):
        print(f"⚠️  Warning: {len(answers)} answers provided for {len(questions)} questions")

    n = min(len(answers), len(questions))
    correct = 0
    category_stats = {}

    print(f"\n{'='*66}")
    print(f"  VEA Benchmark Scoring — {n} questions")
    print(f"{'='*66}\n")

    for i in range(n):
        q = questions[i]
        answer = answers[i]
        expected = q["correct"]
        cat = q.get("category", "Unknown")
        is_correct = answer == expected

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1

        if is_correct:
            correct += 1
            category_stats[cat]["correct"] += 1
            if not errors_only:
                print(f"  Q{q['id']:2d} ✅  {answer}  ({cat})")
        else:
            print(f"  Q{q['id']:2d} ❌  got={answer} expected={expected}  ({cat}/{q.get('subcategory','')})")
            snippet = q["question"][:90]
            print(f"       └─ {snippet}...")

    pct = correct / n * 100
    ci_lo, ci_hi = wilson_ci(correct, n)
    passed = pct >= 70

    print(f"\n{'='*66}")
    print(f"  Score:     {correct}/{n}  ({pct:.1f}%)")
    print(f"  95% CI:    [{ci_lo*100:.1f}% — {ci_hi*100:.1f}%]")
    print(f"  Threshold: ≥70%  →  {'✅ PASSED' if passed else '❌ FAILED'}")
    print()
    print(f"  {'Category':<22} {'Score':>8}  {'Pct':>6}")
    print(f"  {'-'*22} {'-'*8}  {'-'*6}")
    for cat, stats in sorted(category_stats.items()):
        s = f"{stats['correct']}/{stats['total']}"
        cat_pct = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {cat:<22} {s:>8}  {cat_pct:>5.1f}%")
    print(f"{'='*66}\n")


def main():
    parser = argparse.ArgumentParser(description="VEA Benchmark Offline Scorer")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "vea_benchmark_60.json"))
    parser.add_argument("--answers", help="Path to answers file (one letter per line or comma-separated)")
    parser.add_argument("--answers-inline", help="Comma-separated answers string")
    parser.add_argument("--errors-only", action="store_true", help="Show only wrong answers")

    args = parser.parse_args()

    questions = load_answer_key(args.questions)

    if args.answers_inline:
        answers = [a.strip().upper() for a in args.answers_inline.split(",")]
    elif args.answers:
        answers = load_answers_from_file(args.answers)
    else:
        print("Error: provide --answers or --answers-inline")
        sys.exit(1)

    score(questions, answers, errors_only=args.errors_only)


if __name__ == "__main__":
    main()
