# Running Instructions

## Prerequisites

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export GEMINI_API_KEY="your-gemini-api-key"  # same key, needed by litellm for GEPA reflection
```

## Install dependencies

```bash
uv sync
```

## Baseline experiment (original policies)

```bash
# Dry run to see what will be executed
uv run python -m vm.main --dry-run

# Full baseline run (all 7 original policies on 20 videos)
uv run python -m vm.main --concurrency 8

# Run specific policies only
uv run python -m vm.main --policies raw,transcript,summary,mixed --concurrency 8
```

Results are saved to `results/`.

## GEPA-optimized LLM routing

### Step 1: Optimization

Pre-builds materializations (transcripts + summaries) for 10 train videos, then runs GEPA to evolve the routing directive.

```bash
# Recommended: use gemini-2.5-flash for queries during optimization (~2-3 hours)
uv run python -m vm.main --gepa --gepa-max-calls 500 --model gemini-2.5-flash --concurrency 8

# Cheaper/faster test run (seed evaluation only, ~30 min)
uv run python -m vm.main --gepa --gepa-max-calls 3 --concurrency 4

# Use Pro for higher quality optimization (slower, ~10+ hours)
uv run python -m vm.main --gepa --gepa-max-calls 500 --concurrency 4
```

Output: `results/gepa/optimization_result.json` containing the best directive and all candidates.

### Step 2: Evaluation on held-out test set

Evaluates the GEPA-learned directive on 10 test videos alongside all baseline policies. Produces comparison plots.

```bash
# Evaluate with the default model (gemini-3.1-pro-preview)
uv run python -m vm.main --gepa-eval results/gepa/optimization_result.json --concurrency 8

# Or evaluate with Flash for faster results
uv run python -m vm.main --gepa-eval results/gepa/optimization_result.json --model gemini-2.5-flash --concurrency 8
```

Output: `results/gepa_eval/` containing:
- `token_breakdown.csv` — per-policy accuracy and token counts
- `raw_results.json` — detailed predictions
- `tokens_vs_accuracy.png` — total tokens vs accuracy scatter plot
- `tokens_vs_accuracy_query.png` — query tokens vs accuracy scatter plot

## CLI flags reference

| Flag | Default | Description |
|---|---|---|
| `--top-k` | 20 | Number of videos to select |
| `--segment-length` | 30 | Segment length in seconds |
| `--model` | gemini-3.1-pro-preview | Gemini model for materialization and queries |
| `--policies` | raw,transcript,visual-description,summary,low-fps,mixed | Comma-separated policies |
| `--concurrency` | 16 | Max concurrent Gemini API calls |
| `--dry-run` | - | Print plan without making API calls |
| `--gepa` | - | Run GEPA optimization on train split |
| `--gepa-eval PATH` | - | Evaluate a GEPA result on test split |
| `--gepa-max-calls N` | 150 | Override GEPA max metric calls |

## Caching

All API results are cached to disk:
- `data/cache/builders/` — materialization outputs (transcripts, summaries)
- `data/cache/answers/` — QA query responses
- `data/cache/routing/` — LLM routing decisions per (video, directive)
- `data/cache/durations.json` — video duration metadata

Re-running a command will reuse cached results. To force re-computation, delete the relevant cache directory.
