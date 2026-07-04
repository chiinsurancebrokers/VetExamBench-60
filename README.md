# 🩺 VetExamBench-60

**The first open benchmark for veterinary AI knowledge evaluation.**

60 multiple-choice questions with verified answer keys from the official ICVA Veterinary Educational Assessment (VEA®) — the only publicly available, authority-verified veterinary exam dataset.

[![License](https://img.shields.io/badge/Code-Apache%202.0-blue.svg)](LICENSE)
[![Questions](https://img.shields.io/badge/Questions-60-green.svg)]()
[![Source](https://img.shields.io/badge/Source-ICVA%20VEA®-orange.svg)](https://www.icva.net/)

---

## Why This Exists

There is **no open, freely available dataset** of veterinary exam questions with verified answers anywhere online. We searched systematically — GitHub, HuggingFace, PyPI, arXiv — nothing exists. This benchmark fills that gap using the only official free resource: the ICVA VEA® Sample Questions.

## What's Included

```
vea_benchmark_60.json    # 60 questions + answer key + categories + metadata
run_vea_benchmark.py     # Automated runner via Anthropic API
score_answers.py         # Offline scorer for manual grading
LICENSE                  # Apache 2.0 (code only)
NOTICE                   # Third-party content attribution (ICVA)
```

## Question Coverage

| Category | Count | Examples |
|---|---|---|
| Physiology | 12 | Cardiovascular, reproduction, acid-base, ruminant metabolism |
| Pharmacology | 10 | NSAIDs, opioids, antimicrobials, dose calculations |
| Anatomy | 10 | Neuroanatomy, surgical anatomy, equine nerve blocks |
| Microbiology | 7 | Bacteriology, virology, prion diseases, zoonoses |
| Parasitology | 5 | Cestodes, nematodes, equine parasites |
| Clinical Pathology | 5 | Hematology, enzymology, acid-base interpretation |
| Pathology | 4 | Inflammation, oncology, cell adaptation |
| Immunology | 3 | Hypersensitivity, innate immunity, complement |
| Clinical Medicine | 2 | Equine cardiology, neurology |
| Histology | 2 | Epithelium, neurohistology |

> 📷 7 questions reference images (radiographs/photos). Use `--skip-image-questions` for text-only mode (53 questions).

## Quick Start

### Installation

```bash
git clone https://github.com/chiinsurancebrokers/VetExamBench-60.git
cd VetExamBench-60
pip install anthropic   # only needed for API runner
```

### Run Against Claude

```bash
# Full 60-question benchmark
python run_vea_benchmark.py \
    --api-key sk-ant-your-key \
    --model claude-sonnet-4-6

# Text-only mode (53 questions, skip image-dependent)
python run_vea_benchmark.py \
    --api-key sk-ant-your-key \
    --model claude-sonnet-4-6 \
    --skip-image-questions

# Dry run (no API calls — see what would be sent)
python run_vea_benchmark.py --dry-run
```

### Score Pre-Collected Answers

```bash
# Comma-separated answers
python score_answers.py --answers-inline "D,B,A,C,D,B,C,A,A,C,..."

# From a file (one letter per line)
python score_answers.py --answers model_answers.txt

# Show only errors
python score_answers.py --answers model_answers.txt --errors-only
```

## Output

The runner produces a JSON report:

```json
{
  "model": "claude-sonnet-4-6",
  "correct": 47,
  "score_pct": 78.3,
  "ci_95_low": 66.4,
  "ci_95_high": 86.9,
  "passed": true,
  "category_breakdown": {
    "Pharmacology": {"total": 10, "correct": 9, "pct": 90.0},
    "Physiology":   {"total": 12, "correct": 10, "pct": 83.3}
  },
  "wrong_answers": [
    {"id": 3, "category": "Parasitology", "expected": "A", "got": "C"}
  ]
}
```

## Pass Criteria

| Metric | Threshold |
|---|---|
| Overall Score | ≥ 70% (42/60) |
| Confidence Interval | Wilson score 95% CI |

## Methodology

1. **Single-question prompting** — each question is presented individually with a system prompt instructing single-letter responses, minimising prompt engineering effects.
2. **Image handling** — 7 questions reference images. `--skip-image-questions` excludes them; otherwise, `image_note` fields provide textual context.
3. **Confidence intervals** — Wilson score intervals (better calibration than Wald at extreme proportions).
4. **Answer extraction** — regex-based, tolerates parenthesised letters, full words, and extraneous text.

## Integration with PetAiNurse

This benchmark serves as **Tier 1** in the PetAiNurse three-tier veterinary AI evaluation:

| Tier | Benchmark | What it tests | Pass |
|---|---|---|---|
| 0 | VetTriageBench-45 | Triage safety — 0% unsafe undertriage | ✅ |
| **1** | **VetExamBench-60** | **Basic science knowledge (vet school level)** | **≥70%** |
| 2 | VetLicenseBench-60 | Clinical licensing knowledge (NAVLE level) | ≥60% |

## License

- **Code** (`run_vea_benchmark.py`, `score_answers.py`): [Apache License 2.0](LICENSE)
- **Question content** (`vea_benchmark_60.json`): © [ICVA](https://www.icva.net/). Included for non-commercial evaluation under fair use. See [NOTICE](NOTICE).

## Citation

If you use this benchmark in research or publications:

```bibtex
@misc{vetexambench60,
  title={VetExamBench-60: An Open Benchmark for Veterinary AI Knowledge Evaluation},
  author={KiraAIPet},
  year={2026},
  howpublished={\url{https://github.com/chiinsurancebrokers/VetExamBench-60}},
  note={Based on ICVA VEA® Sample Questions}
}
```

## Contributing

PRs welcome for:
- Additional API adapters (OpenAI, Google, etc.)
- Localised question translations
- Analysis scripts and visualisations

**Do NOT** submit additional copyrighted exam questions without proper licensing.
