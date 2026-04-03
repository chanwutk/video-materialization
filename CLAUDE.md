# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating how static segment-level materialization policies trade off token usage and answer accuracy for repeated-query video QA workloads. Videos are segmented into fixed-length parts, each materialized before seeing queries, and the same policy is reused for all questions on that video.

**Datasets:** ActivityNet-QA (primary), Neptune (secondary)

**Materialization types:** transcript, sampled keyframes, short summary

**Policies under comparison:**
- Raw baseline (full video)
- Transcript-only
- Keyframes-only
- Summary-only
- Three-tier mixed (transcript for speech-dense, keyframes for visually active, summary for low-information segments)

**Model:** Gemini (with explicit thinking-token logging)

**Accuracy metric:** benchmark-native deterministic matching when available; LLM judge fallback

## Key Design Decisions

- Materialization policy is query-independent
- Token accounting separates build-time vs query-time costs
- Embeddings are out of scope
- See PLAN.md for full experiment plans and implementation checklist
