# Evaluation Harness for an OCR-to-Explanation Math-Solver Pipeline

> Companion evaluation repository for [AIMS](https://github.com/osten-antonio/AIMS)

## Overview

AIMS (AI-Powered Math Solver) is a self-hostable, end-to-end pipeline for AI-assisted mathematical problem solving. The application combines an OCR-free vision transformer for direct image-to-LaTeX ingestion with a two-stage solver (a deterministic symbolic engine followed by a swappable LLM backend) and an explanation service that produces hints, step-by-step solutions, and practice questions using an answer-first reasoning strategy. The full stack is containerized via Docker, uses Redis for response caching, and is hardened against common prompt-injection and XSS patterns.

This repository contains **only the evaluation tooling**; the scripts, datasets, and notebook used to reproduce the experimental results reported in the paper. The application itself lives in the main [AIMS](https://github.com/osten-antonio/AIMS) repository.

The harness covers four evaluation axes:

- **OCR accuracy** - benchmarks three image-to-LaTeX models (Nougat-base, the fine-tuned `Norm/nougat-latex-base` checkpoint, and pix2tex) on a synthetic benchmark rendered from LaTeX, plus a zero-shot probe on a real-world handwritten-equation dataset.
- **Solver accuracy** - drives the deployed solver API against a 60-item QA dataset spanning arithmetic, algebra, geometry, percentages, unit conversion, probability, sequences, systems of equations, and multi-step word problems, across multiple locally-deployed LLM backends.
- **Explanation quality** - generates anchored vs. unanchored step-by-step solutions on an ASyMOB subset to empirically validate the answer-first reasoning strategy, with a SymPy-based checker isolating arithmetic and phantom-number errors and a McNemar exact test for significance.
- **Cache effectiveness** - measures cold-call vs. cache-hit latency on the live solver endpoint and reads server-side cache statistics to separate genuine inference latency from request-handling overhead.

## Repository layout

```
AIMS-eval/
├── evaluate_ocr.py            # OCR accuracy benchmark (Nougat-base / Nougat-latex-base / pix2tex)
├── evaluate_solver.py         # Solver accuracy benchmark against a 60-item QA set
├── evaluate_explanation.py    # Answer-first reasoning evaluation (anchored vs. unanchored, ASyMOB)
├── evaluate_cache.py          # Cache cold-call vs. cache-hit latency benchmark
├── generate_equations.py       # Synthesizes the LaTeX equation corpus used by the OCR benchmark
├── render_items.py             # Renders LaTeX equations to PNG (printed + handwritten styles)
├── generate_steps.py           # Calls the AIMS explanation service to produce anchored/unanchored steps
├── strip_200_unique.py         # Samples a 200-item reproducible ASyMOB subset for the explanation eval
├── evaluation_summary.ipynb    # Aggregates all CSV outputs into the figures and tables from the paper
├── ocr-server/                 # Standalone FastAPI server exposing all three OCR models via HTTP
│   ├── server.py
│   └── pyproject.toml
├── data/
│   └── qa_pairs.json           # 60-item solver QA dataset (committed; larger datasets are gitignored)
├── pyproject.toml               # Python ≥ 3.13, managed via `uv`
└── uv.lock
```

## Dependencies

The evaluation harness requires Python ≥ 3.13 and the packages listed in `pyproject.toml`:

- `requests`: HTTP client for driving the AIMS API and the OCR server
- `pandas`: tabular output for per-question and aggregated results
- `matplotlib`: figures produced inside `evaluation_summary.ipynb`
- `jiwer`: character-level Levenshtein / normalized edit distance for OCR scoring
- `fonttools`: font handling during the LaTeX-to-image rendering stage
- `opencv-python` (`cv2`): required by `evaluate_ocr.py` for image cropping on the real-world dataset (imported lazily, so install only if you run the OCR benchmark)
- `pix2tex`, `transformers`, `torch`, `fastapi`, `uvicorn`, `pillow`: only required if you run the local OCR server (`ocr-server/`)
- Seperate LLM backend if you are self-hosting the models, this repo uses Ollama

## Prerequisites

The evaluation scripts assume two services are reachable over HTTP:

1. **The AIMS application** (solver, explanation, and practice endpoints). It must be running and serving, by default, at `http://localhost:8000`. Clone and start it from the evaluation branch, [https://github.com/osten-antonio/AIMS/tree/evaluation](https://github.com/osten-antonio/AIMS/tree/evaluation). The default routes the harness expects are:
   - `POST /solver/solve`
   - `POST /explanation/steps/`
   - `GET  /solver/cache-stats`
   - `POST /solver/cache-reset`
2. **The OCR server** (only needed for `evaluate_ocr.py`). A small FastAPI wrapper around the three OCR models, served by default at `http://localhost:9000` with endpoints `/nougat-base/inference`, `/nougat-latex-base/inference`, and `/pix2tex/inference`. Source lives in `ocr-server/`.

If your AIMS instance or OCR server runs on a different host or port, override the defaults via the `--url` flag on each script.

## Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) for dependency management. From the repository root:

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the environment and install dependencies
uv sync

# 3. Activate the virtual environment (or prefix every command with `uv run`)
source .venv/bin/activate
```

To evaluate the OCR performance, run the OCR server, install its dependencies separately:

```bash
cd ocr-server
uv sync
cd ..
```

## Usage

The harness is a collection of independent CLI scripts. Each writes its outputs to a `results/` directory (created on demand) as CSV files named after the model under test. Run them in any order; the notebook at the end aggregates everything.

### 1. Generate the synthetic OCR benchmark

`generate_equations.py` synthesizes a corpus of LaTeX equations and word problems across categories. `render_items.py` then renders each equation to PNG in two styles, **printed** (via `pdflatex`) and **handwritten** (via `xelatex` with handwriting-style TTFs), and writes a manifest.

```bash
# Produce the LaTeX corpus + manifest (paths configurable inside the script)
uv run python generate_equations.py

# Render the manifest into images (requires pdflatex / xelatex on PATH)
uv run python render_items.py
```

Place the resulting `data/images/`, `data/tex/`, and `data/manifest.json` under `data/` (these are gitignored).

### 2. Start the OCR server

```bash
cd ocr-server
uv run uvicorn server:app --host 0.0.0.0 --port 9000
```

### 3. Run the OCR accuracy benchmark

```bash
# Synthetic rendered benchmark (printed + handwritten styles)
uv run python evaluate_ocr.py \
  --manifest data/manifest.json \
  --image-root data/images \
  --output results/ocr_synthetic.csv
```

The script reports Levenshtein distance, normalized edit distance, and token accuracy per (model, style) cell, mirroring the OCR results table in the paper.

For the real-world zero-shot probe (FASTMATHxASSUMPTION 2025), point `--manifest`, `--image-root`, and `--output` at the FASTMATH layout instead. The FASTMATH source CSV and image root are gitignored due to licensing; download them from [Kaggle](https://www.kaggle.com/datasets/mssjss/fastmathxassumption-2025-handwritten-math-to-latex).

### 4. Start the AIMS application

Follow the instructions in [github.com/osten-antonio/AIMS](https://github.com/osten-antonio/AIMS) to launch the backend (via npm run dev or docker).

### 5. Run the solver accuracy benchmark

`evaluate_solver.py` posts each question in `data/qa_pairs.json` to `/solver/solve`, extracts the numerical answer from the response, and compares it to the expected value with a tolerance that scales with magnitude (1% relative for medium numbers, 5% for large, 0.1 absolute for small decimals).

```bash
uv run python evaluate_solver.py \
  --model gemma-4-12b-qat:latest \
  --data data/qa_pairs.json \
  --url http://localhost:8000/solver/solve \
  --output-dir results
```

Repeat for every LLM backend you want to compare. Outputs:

- `results/output_<model>.csv`: per-question answers and correctness flags
- `results/summary_<model>.csv`: aggregate accuracy

### 6. Run the cache effectiveness benchmark

For each question, `evaluate_cache.py` issues the same solve request twice (first as a cold call then as a cache hit) and records both latencies plus the speedup ratio. It also pulls server-side cache statistics to separate genuine inference time from request-handling overhead.

```bash
uv run python evaluate_cache.py \
  --model gemma-4-12b-qat:latest \
  --data data/qa_pairs.json \
  --url http://localhost:8000/solver/solve \
  --output-dir results
```

### 7. Run the explanation (answer-first reasoning) evaluation

This is a two-step process.

**Step 7a. Prepare the ASyMOB subset.** `strip_200_unique.py` reads the full ASyMOB dataset (download separately; see paper citation), drops the "Original" variations, deduplicates by challenge text, and reproducibly samples 200 items (seed = 5, two per derivation family) covering all 100 derivations.

```bash
# Place Full_ASyMOB_Dataset.json under data/ first (gitignored)
uv run python strip_200_unique.py
# writes to data/ASyMOB_200_unique.json
```

**Step 7b. Generate anchored vs. unanchored steps.** For each item, `generate_steps.py` calls `/explanation/steps/` twice: once with the correct final answer supplied (anchored), once with an empty answer (unanchored). Both raw step lists are serialized to CSV.

```bash
uv run python generate_steps.py \
  --data data/ASyMOB_200_unique.json \
  --model gemma-4-12b-qat:latest \
  --limit 200
# writes to results/explanations_ASyMOB_<model>.csv
```

**Step 7c. Score with the checker and judge.** `evaluate_explanation.py` runs the SymPy-based deterministic checker (which isolates arithmetic errors and phantom numbers) and a judge model for semantic correctness (Qwen3.6-3.5B-a3B was used for running the evaluations here), then runs McNemar's exact test on the per-question correctness vector. See the script's `--help` for the full flag list; the defaults match the configuration reported in the paper. 

### 8. Aggregate everything

`evaluation_summary.ipynb` reads every CSV under `results/` and reproduces the figures and tables reported in the paper (OCR metric tables, solver accuracy comparison, cache speedup plot, anchored vs. unanchored delta).

```bash
uv run jupyter lab evaluation_summary.ipynb
```

## Results

The numbers below summarize the empirical findings reproduced by this harness on the hardware described in the paper (32 GB RAM, RTX 3060, AMD Ryzen 5 5600G; Redis and the model server co-located on the same host).

- **OCR (synthetic, 200 images)**: the fine-tuned `Norm/nougat-latex-base` checkpoint achieves the lowest normalized edit distance, with pix2tex showing a roughly 21% hard-failure rate (no valid output at all). Vanilla Nougat leads on a few individual metrics but hallucinates academic-paper-style prose on real-world input.
- **OCR (zero-shot, FASTMATH subset, n=2,308)**: all three models exhibit substantially higher edit distances than on the synthetic benchmark; the fine-tuned checkpoint is retained for deployment as it remains the best trade-off on real data.
- **Solver accuracy (60-item QA set, six models)**: accuracy ranges from ~70% (`gemma-4:E2B-it`) to ~88.3% (`gemma-4:12b-qat`), an ~18-point spread. One additional model (`qwen3.5:9b`) was excluded from the comparison due to a schema-conformance bug unrelated to underlying capability.
- **Answer-first reasoning (ASyMOB 200-item subset)**: anchoring on the correct final answer lifts the weakest model from 9.5% to 34.5% accuracy (p ≈ 1×10⁻¹¹) and the strongest from 37% to 79.5% (p ≈ 1.7×10⁻¹⁹), with arithmetic error rates roughly halved on the smaller model. Phantom-number rates improve on the larger model and regress negligibly on the smaller one; hallucination is reduced, not eliminated.
- **Cache effectiveness**: median speedup of ~700× on the 12B model (1313.69 ms cold vs. 1.9 ms hit) and ~400× on the E2B model (735.79 ms cold vs. 1.89 ms hit); server-side p99 cache-hit latency stays under 1 ms.

Refer to `evaluation_summary.ipynb` for the full per-model tables and figures.

## Datasets

| Purpose | Dataset | Source |
|---|---|---|
| Synthetic OCR benchmark | Rendered in-repo by `generate_equations.py` + `render_items.py` | This repo |
| Real-world OCR probe | FASTMATHxASSUMPTION 2025 (filtered to n=2,308 valid outputs) | [Kaggle](https://www.kaggle.com/datasets/mssjss/fastmathxassumption-2025-handwritten-math-to-latex) |
| Solver QA | 60 hand-authored items across 9 categories | `data/qa_pairs.json` |
| Explanation eval | ASyMOB (200-item reproducible subset) | [arXiv:2505.23851](https://arxiv.org/abs/2505.23851) |

### Notes
- All LLM calls in the test suite use temperature 0 for determinism; the answer-first evaluation runs are also at temperature 0.
