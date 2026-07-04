#!/usr/bin/env python3
"""
VetExamBench-60 — KiraAIPet Live Endpoint Runner
==================================================
Tests your deployed PetAiNurse app (not the raw model).
Benchmarks the full stack: system prompt + veterinary context + model.

Usage:
    python run_kiraaipet_benchmark.py
    python run_kiraaipet_benchmark.py --url https://petainurse.up.railway.app
    python run_kiraaipet_benchmark.py --skip-image-questions
    python run_kiraaipet_benchmark.py --endpoint /api/chat

Requirements:
    pip install requests
"""

import json
import math
import re
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: pip install requests")
    sys.exit(1)

DEFAULT_URL = "https://petainurse.up.railway.app"
ENDPOINTS_TO_TRY = ["/api/chat", "/api/message", "/api/ai", "/api/ask"]


def wilson_ci(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    m = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / d
    return (max(0, c-m), min(1, c+m))


def load_questions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_endpoint(base_url):
    print(f"  🔍 Discovering endpoint at {base_url} ...")
    try:
        r = requests.get(base_url, timeout=10)
        print(f"     App is up (HTTP {r.status_code})")
    except Exception as e:
        print(f"     ⚠️  App not responding: {e}")
        sys.exit(1)

    test = {"messages": [{"role": "user", "content": "hello"}]}
    for ep in ENDPOINTS_TO_TRY:
        try:
            r = requests.post(f"{base_url}{ep}", json=test, timeout=15,
                              headers={"Content-Type": "application/json"})
            if r.status_code == 200:
                print(f"     ✅ Found: {ep}")
                return ep
            print(f"     ❌ {ep} → HTTP {r.status_code}")
        except requests.exceptions.Timeout:
            print(f"     ⏱️  {ep} → timeout (likely streaming — using it)")
            return ep
        except Exception as e:
            print(f"     ❌ {ep} → {e}")

    print(f"\n  Could not find endpoint. Use --endpoint /api/chat")
    sys.exit(1)


def call_app(base_url, endpoint, question, timeout=30):
    prompt = (
        "I'm going to ask you a veterinary exam question. "
        "Reply with ONLY the letter of the correct answer (A, B, C, D, E, or F). "
        f"No explanation — just the letter.\n\n{question}"
    )
    body = {"messages": [{"role": "user", "content": prompt}]}

    try:
        r = requests.post(f"{base_url}{endpoint}", json=body, timeout=timeout,
                          headers={"Content-Type": "application/json"}, stream=True)
        r.raise_for_status()
        ct = r.headers.get("content-type", "")

        if "application/json" in ct:
            d = r.json()
            if isinstance(d, dict):
                for key in ["content", "response", "text"]:
                    if key in d: return d[key]
                if "message" in d and isinstance(d["message"], dict):
                    return d["message"].get("content", "")
                if "choices" in d and d["choices"]:
                    return d["choices"][0].get("message", {}).get("content", "")
            return str(d)
        else:
            txt = ""
            for chunk in r.iter_content(decode_unicode=True):
                if chunk: txt += chunk
            txt = re.sub(r'^data:\s*', '', txt, flags=re.MULTILINE)
            return txt.replace('[DONE]', '').strip()
    except Exception as e:
        return f"ERROR: {e}"


def extract_answer(raw):
    raw = raw.strip()
    if len(raw) > 10:
        for pat in [r'(?:answer|correct)\s+(?:is|:)\s*\(?([A-F])\)?',
                     r'\*\*([A-F])\*\*', r'^\(?([A-F])\)?[\.\s]',
                     r'\(?([A-F])\)?\s*$']:
            m = re.search(pat, raw, re.IGNORECASE | re.MULTILINE)
            if m: return m.group(1).upper()
        m = re.search(r'\b([A-F])\b', raw)
        if m: return m.group(1).upper()
    u = raw.upper().strip()
    if len(u) == 1 and u in "ABCDEF": return u
    m = re.search(r'\(?([A-F])\)?', u)
    return m.group(1) if m else (u[:1] if u else "?")


def build_question(q):
    lines = [f"Question {q['id']}: {q['question']}", ""]
    for letter, text in q["options"].items():
        lines.append(f"({letter}) {text}")
    if q.get("requires_image") and q.get("image_note"):
        lines.append(f"\n[Image context: {q['image_note']}]")
    return "\n".join(lines)


def run(base_url, endpoint, qpath, skip_image=False, delay=1.5):
    data = load_questions(qpath)
    meta = data["metadata"]
    questions = data["questions"]
    if skip_image:
        questions = [q for q in questions if not q.get("requires_image")]

    total = len(questions)
    correct = 0
    wrong = []
    results = []
    cats = {}

    print(f"\n{'='*66}")
    print(f"  VetExamBench-60 → KiraAIPet Live")
    print(f"  Target:    {base_url}{endpoint}")
    print(f"  Questions: {total}")
    print(f"{'='*66}\n")

    for i, q in enumerate(questions, 1):
        prompt = build_question(q)
        cat = q.get("category", "Unknown")
        if cat not in cats:
            cats[cat] = {"total": 0, "correct": 0, "wrong_ids": []}
        cats[cat]["total"] += 1

        raw = call_app(base_url, endpoint, prompt)
        ans = extract_answer(raw)
        ok = ans == q["correct"]

        if ok:
            correct += 1
            cats[cat]["correct"] += 1
            print(f"  [{i:2d}/{total}] Q{q['id']:2d}  ✅  app={ans}  key={q['correct']}  ({cat})")
        else:
            wrong.append({"id": q["id"], "category": cat,
                          "expected": q["correct"], "got": ans,
                          "raw": raw[:200],
                          "requires_image": q.get("requires_image", False)})
            cats[cat]["wrong_ids"].append(q["id"])
            print(f"  [{i:2d}/{total}] Q{q['id']:2d}  ❌  app={ans}  key={q['correct']}  ({cat})")

        results.append({"id": q["id"], "app_answer": ans,
                        "correct": q["correct"], "is_correct": ok, "category": cat})
        if i < total: time.sleep(delay)

    pct = (correct/total*100) if total else 0
    lo, hi = wilson_ci(correct, total)
    passed = pct >= meta.get("pass_threshold_pct", 70)

    print(f"\n{'='*66}")
    print(f"  🩺 KiraAIPet RESULTS")
    print(f"{'='*66}")
    print(f"  Score:     {correct}/{total}  ({pct:.1f}%)")
    print(f"  95% CI:    [{lo*100:.1f}% — {hi*100:.1f}%]")
    print(f"  Verdict:   {'✅ PASSED' if passed else '❌ FAILED'}")
    print()
    for cat, s in sorted(cats.items()):
        cp = s["correct"]/s["total"]*100 if s["total"] else 0
        print(f"  {cat:<22} {s['correct']}/{s['total']:>2}  {cp:>5.1f}%")
    if wrong:
        print(f"\n  Wrong ({len(wrong)}):")
        for w in wrong:
            print(f"    Q{w['id']:2d} expected={w['expected']} got={w['got']}")
    print(f"{'='*66}\n")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target": f"{base_url}{endpoint}", "test_type": "kiraaipet_live",
        "correct": correct, "total": total, "score_pct": round(pct, 1),
        "ci_95": [round(lo*100,1), round(hi*100,1)], "passed": passed,
        "wrong_answers": wrong, "results": results,
    }
    return report


def main():
    p = argparse.ArgumentParser(description="VetExamBench-60: KiraAIPet Live Test")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--endpoint", default=None)
    p.add_argument("--questions", default=str(Path(__file__).parent / "vea_benchmark_60.json"))
    p.add_argument("--skip-image-questions", action="store_true")
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    base = args.url.rstrip("/")
    ep = args.endpoint or discover_endpoint(base)

    report = run(base, ep, args.questions, args.skip_image_questions, args.delay)

    out = args.output or f"kiraaipet_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Report saved → {out}")


if __name__ == "__main__":
    main()
