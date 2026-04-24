# Video Materialization — Experiment Plan

## Core Question

For repeated-query video workloads, how do segment-level materialization policies trade off token cost and answer accuracy?

## Dataset

MINERVA Video QA benchmark. 20 videos selected (10 questions each, 200 total). Videos 293s–1389s. 10/10 train/test split (seed 42).

## Policies Compared

| Policy | How it works | Cost |
|---|---|---|
| **raw** | Full video at default FPS | Highest |
| **low-fps** | Video at 0.2 FPS | High |
| **low-res** | Low-resolution video | High |
| **transcript** | Per-segment speech transcription → text | Low |
| **summary** | Whole-video text summary | Lowest |
| **visual-description** | Whole-video scene descriptions → text | Low |
| **mixed** | Hand-tuned heuristic: word-count thresholds decide transcript/summary/low-fps per segment | Medium |
| **llm-routed** | GEPA-learned LLM routing: evolved directive decides per-segment materialization | Medium |

## What's New for the Final Report

### Problem with `mixed` (reviewer feedback)
The 3-tier heuristic (transcript words > 30 → transcript; speech or summary words > 50 → low-fps; else → summary) is unprincipled. Accuracy was also mediocre.

### Solution: GEPA-optimized LLM routing (`llm-routed`)
Replace the hand-tuned thresholds with a **learned routing directive**, evolved via GEPA (evolutionary prompt optimization).

**How it works:**
1. An LLM router sees each segment's transcript + summary text (pre-computed, cached)
2. The router follows a **policy directive** (natural language instructions) to decide: {TRANSCRIPT, SUMMARY, LOW_FPS, SKIP} per segment
3. GEPA evolves the directive text by evaluating candidate directives on train videos and reflecting on failures

**Key design choices:**
- Router model: `gemini-2.5-flash-lite` (cheapest, text-only input)
- GEPA reflection: `gemini-2.5-flash` (proposes directive improvements)
- Query model: `gemini-2.5-flash` during optimization, `gemini-3.1-pro-preview` for final eval
- Routing is per-video (same decisions for all questions on a video)
- Fixed prompt template with evolvable directive section — GEPA only mutates the decision guidelines
- Models without thinking support (Flash) automatically use configs without ThinkingConfig

**Evaluation:**
- GEPA optimizes on 10 train videos (100 questions), Pareto-aware candidate selection
- Final evaluation on 10 held-out test videos alongside all baselines
- Comparison: accuracy vs query token cost scatter plot (same format as preliminary report)

## Running

```bash
# GEPA optimization (~2-3 hours on Flash)
uv run python -m vm.main --gepa --gepa-max-calls 500 --model gemini-2.5-flash --concurrency 8

# Evaluate on test set
uv run python -m vm.main --gepa-eval results/gepa/optimization_result.json --concurrency 8
```

See `RUN.md` for full details. Remote deployment scripts in `scripts/`.

## Status

- [x] Baseline policies implemented and evaluated (preliminary report)
- [x] LLM router with GEPA integration implemented
- [x] Train/test split, caching, model-aware configs
- [x] Sanity check passed: 53% seed accuracy on train set, pipeline end-to-end verified
- [ ] Full GEPA optimization run (500 metric calls)
- [ ] Test set evaluation with learned directive vs baselines
- [ ] Final report figures and writeup
