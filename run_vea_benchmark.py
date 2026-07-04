#!/usr/bin/env python3
"""
VetExamBench-60 — Multi-Model Benchmark Runner
================================================
Tests Claude AND GPT models against 60 ICVA VEA questions.

Usage:
    # Claude
    python run_vea_benchmark.py --provider anthropic --api-key sk-ant-... --model claude-sonnet-4-6

    # GPT-4o
    python run_vea_benchmark.py --provider openai --api-key sk-... --model gpt-4o

    # GPT-o3
    python run_vea_benchmark.py --provider openai --api-key sk-... --model o3

    # Text-only (skip 7 image questions)
    python run_vea_benchmark.py --provider anthropic --api-key sk-ant-... --skip-image-questions

    # Dry run
    python run_vea_benchmark.py --dry-run

Requirements:
    pip install requests

Source: ICVA VEA® Sample Questions — © ICVA. Used for non-commercial evaluation.
"""

import json
import os
import re
import sys
import math
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Wilson score 95% CI
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
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
# Prompt
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
    if q.get("requires_image") and q.get("image_note"):
        lines.append(f"\n[Image context: {q['image_note']}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API Callers
# ---------------------------------------------------------------------------

def call_anthropic(api_key: str, model: str, prompt: str) -> str:
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 16,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


def call_openai(api_key: str, model: str, prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 16,
        "temperature": 0,
    }
    # o-series models don't support system messages or temperature
    if model.startswith(("o1", "o3", "o4")):
        body["messages"] = [
            {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt}
        ]
        del body["temperature"]
        del body["max_tokens"]
        body["max_completion_tokens"] = 64

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers, json=body, timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def call_model(provider: str, api_key: str, model: str, prompt: str) -> str:
    if provider == "anthropic":
        return call_anthropic(api_key, model, prompt)
    elif provider == "openai":
        return call_openai(api_key, model, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Extract answer letter
# ---------------------------------------------------------------------------

def extract_answer(raw: str) -> str:
    raw = raw.strip()
    if len(raw) > 10:
        for pat in [
            r'(?:answer|correct)\s+(?:is|:)\s*\(?([A-F])\)?',
            r'\*\*([A-F])\*\*',
            r'^\(?([A-F])\)?[\.\s]',
            r'\(?([A-F])\)?\s*$',
        ]:
            m = re.search(pat, raw, re.IGNORECASE | re.MULTILINE)
            if m:
                return m.group(1).upper()
    raw_upper = raw.upper().strip()
    if len(raw_upper) == 1 and raw_upper in "ABCDEF":
        return raw_upper
    m = re.search(r'\(?([A-F])\)?', raw_upper)
    if m:
        return m.group(1)
    return raw_upper[:1] if raw_upper else "?"


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def run_benchmark(provider, api_key, model, questions_path,
                  skip_image=False, dry_run=False, delay=0.5):
    data = load_questions(questions_path)
    meta = data["metadata"]
    questions = data["questions"]

    if skip_image:
        questions = [q for q in questions if not q.get("requires_image")]

    total = len(questions)
    correct = 0
    wrong = []
    results = []
    category_stats = {}
    label = f"{provider}/{model}"

    print(f"\n{'='*66}")
    print(f"  VetExamBench-60  |  {label}")
    print(f"  Questions: {total}  |  {'TEXT-ONLY' if skip_image else 'FULL'}")
    print(f"{'='*66}\n")

    for i, q in enumerate(questions, 1):
        prompt = build_user_prompt(q)
        cat = q.get("category", "Unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0, "wrong_ids": []}
        category_stats[cat]["total"] += 1

        if dry_run:
            print(f"  [{i:2d}/{total}] Q{q['id']:2d}  ({cat})  — DRY RUN")
            results.append({"id": q["id"], "model_answer": "?",
                            "correct_answer": q["correct"],
                            "is_correct": False, "category": cat})
            continue

        try:
            raw = call_model(provider, api_key, model, prompt)
            model_answer = extract_answer(raw)
        except Exception as e:
            print(f"  [{i:2d}/{total}] Q{q['id']:2d}  ⚠️  ERROR: {e}")
            model_answer = "?"
            raw = str(e)
            time.sleep(5)

        is_correct = model_answer == q["correct"]
        if is_correct:
            correct += 1
            category_stats[cat]["correct"] += 1
            mark = "✅"
        else:
            wrong.append({"id": q["id"], "category": cat,
                          "subcategory": q.get("subcategory", ""),
                          "expected": q["correct"], "got": model_answer,
                          "raw_response": raw[:200],
                          "requires_image": q.get("requires_image", False)})
            category_stats[cat]["wrong_ids"].append(q["id"])
            mark = "❌"

        print(f"  [{i:2d}/{total}] Q{q['id']:2d}  {mark}  model={model_answer}  key={q['correct']}  ({cat})")
        results.append({"id": q["id"], "model_answer": model_answer,
                        "correct_answer": q["correct"],
                        "is_correct": is_correct, "category": cat})

        if i < total:
            time.sleep(delay)

    pct = (correct / total * 100) if total > 0 else 0
    ci_lo, ci_hi = wilson_ci(correct, total)
    passed = pct >= meta.get("pass_threshold_pct", 70)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider, "model": model,
        "source": meta.get("title", "VEA Benchmark"),
        "total_questions": total,
        "image_questions_skipped": skip_image,
        "correct": correct, "wrong_count": total - correct,
        "score_pct": round(pct, 1),
        "ci_95_low": round(ci_lo * 100, 1),
        "ci_95_high": round(ci_hi * 100, 1),
        "pass_threshold_pct": meta.get("pass_threshold_pct", 70),
        "passed": passed,
        "category_breakdown": {},
        "wrong_answers": wrong,
        "results_per_question": results,
    }
    for cat, s in sorted(category_stats.items()):
        cp = (s["correct"] / s["total"] * 100) if s["total"] > 0 else 0
        report["category_breakdown"][cat] = {
            "total": s["total"], "correct": s["correct"],
            "pct": round(cp, 1), "wrong_question_ids": s["wrong_ids"]}

    # Summary
    print(f"\n{'='*66}")
    print(f"  RESULTS — {label}")
    print(f"{'='*66}")
    print(f"  Score:     {correct}/{total}  ({pct:.1f}%)")
    print(f"  95% CI:    [{ci_lo*100:.1f}% — {ci_hi*100:.1f}%]")
    print(f"  Threshold: ≥{meta.get('pass_threshold_pct', 70)}%")
    print(f"  Verdict:   {'✅ PASSED' if passed else '❌ FAILED'}")
    print()
    print(f"  {'Category':<22} {'Score':>8}  {'Pct':>6}")
    print(f"  {'-'*22} {'-'*8}  {'-'*6}")
    for cat, s in sorted(report["category_breakdown"].items()):
        print(f"  {cat:<22} {s['correct']}/{s['total']:>2}      {s['pct']:>5.1f}%")
    if wrong:
        print(f"\n  Wrong ({len(wrong)}):")
        for w in wrong:
            img = " 📷" if w["requires_image"] else ""
            print(f"    Q{w['id']:2d} [{w['category']}] expected={w['expected']} got={w['got']}{img}")
    print(f"{'='*66}\n")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="VetExamBench-60: Multi-Model Runner (Claude + GPT)")
    parser.add_argument("--provider", choices=["anthropic", "openai"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--questions",
                        default=str(Path(__file__).parent / "vea_benchmark_60.json"))
    parser.add_argument("--skip-image-questions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.dry_run:
        args.provider = args.provider or "anthropic"
        api_key = "dry-run"
    else:
        if not args.provider:
            print("Error: --provider anthropic or --provider openai required")
            sys.exit(1)
        api_key = args.api_key or os.environ.get(
            "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY")
        if not api_key:
            env = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
            print(f"Error: --api-key or {env} env var required")
            sys.exit(1)

    if not args.model:
        args.model = {"anthropic": "claude-sonnet-4-6",
                      "openai": "gpt-4o"}[args.provider]

    report = run_benchmark(
        args.provider, api_key, args.model, args.questions,
        args.skip_image_questions, args.dry_run, args.delay)

    if not args.dry_run:
        out = args.output or f"report_{args.provider}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  Report saved → {out}")


if __name__ == "__main__":
    main()
