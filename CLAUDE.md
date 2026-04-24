# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating how segment-level materialization policies trade off token usage and answer accuracy for repeated-query video QA workloads. Videos are segmented into 30-second parts, each materialized before seeing queries, and the same policy is reused for all questions on that video.

**Paper repo:** `./paper` (symlink to `../materialized-video`)

## Architecture

```
src/vm/
  main.py           # CLI entrypoint and experiment orchestrator
  config.py          # Constants, paths, model names, thresholds
  dataset.py         # MINERVA dataset loading, grouping, train/test split
  segmenter.py       # Fixed-length video segmentation (30s)
  duration.py        # YouTube video duration fetching via yt-dlp
  builders.py        # Async Gemini calls to generate transcripts/summaries
  policies.py        # Policy enum, materialization dispatch, SegmentMaterial
  runner.py          # QA prompt construction and answer parsing
  evaluator.py       # Deterministic accuracy matching
  tokens.py          # TokenUsage / PolicyTokenLog dataclasses
  cache.py           # Disk-based caching for builders and answers
  genai_config.py    # Gemini API config (temperature, schema, safety)
  genai_response.py  # Split Gemini response into main text + thoughts
  router.py          # LLM-based segment routing (GEPA-optimized directive)
  gepa_optimizer.py  # GEPA integration for routing policy optimization
```

## Policies Under Comparison

| Policy | Description |
|---|---|
| `raw` | Full video at default FPS |
| `transcript` | Per-segment speech transcription |
| `visual-description` | Whole-video scene descriptions |
| `summary` | Whole-video text summary |
| `low-fps` | Video at 0.2 FPS |
| `low-res` | Low-resolution video |
| `mixed` | 3-tier heuristic routing (word-count thresholds) |
| `llm-routed` | GEPA-optimized LLM routing per segment |

## GEPA-Optimized LLM Routing

The `llm-routed` policy replaces the hand-tuned `mixed` heuristic with a learned routing directive. GEPA (evolutionary prompt optimization) evolves a policy directive that an LLM uses to decide per-segment materialization:

- **Router model:** `gemini-2.5-flash-lite` (cheapest, text-only)
- **Reflection model:** `gemini-2.5-flash` (via litellm)
- **Query model:** configurable, default `gemini-3.1-pro-preview`
- **Routing decisions:** {TRANSCRIPT, SUMMARY, LOW_FPS, SKIP} per segment
- **GEPA objectives:** Pareto-aware selection optimizing accuracy
- **Train/test split:** 10/10 videos, seed 42

## Running Experiments

See `RUN.md` for full instructions. Quick reference:

```bash
# Baseline experiment
uv run python -m vm.main --concurrency 8

# GEPA optimization
uv run python -m vm.main --gepa --gepa-max-calls 500 --model gemini-2.5-flash --concurrency 8

# Evaluate GEPA result on test set
uv run python -m vm.main --gepa-eval results/gepa/optimization_result.json --concurrency 8
```

Remote deployment scripts are in `scripts/`.

## Key Design Decisions

- Materialization policy is per-video (query-independent)
- Token accounting separates build-time vs query-time costs
- All API results are cached to disk under `data/cache/`
- GEPA evolves only the policy directive section within a fixed prompt template
- Answer cache keys include routing_hash for LLM_ROUTED to prevent cross-directive pollution
- See `PLAN.md` for detailed experiment plans and implementation history

## Environment

- Python 3.12+
- Requires `GOOGLE_API_KEY` and `GEMINI_API_KEY` environment variables
- Dependencies: google-genai, gepa, litellm, pandas, altair, rich, yt-dlp
