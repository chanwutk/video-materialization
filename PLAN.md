# Video Materialization

# Video Materialization

## Scoped prelim version

This page tracks the narrowed preliminary-report scope only.

## Locked decisions

- primary dataset: ActivityNet-QA
- second dataset: Neptune
- task: repeated video QA with multiple questions per video
- materialization policy is query-independent
- fixed-length segmentation in the first version
- materialization menu: transcript, keyframes, summary
- mixed policy: three-tier transcript / keyframes / summary
- embeddings are out of scope
- use Gemini and log thinking tokens explicitly
- accuracy metric: benchmark-native deterministic matching when available; otherwise LLM judge

## Core question

For repeated-query video workloads, how do static segment-level materialization policies trade off token usage and answer accuracy?

## Design setup

A video is segmented into fixed-length parts. Each segment is materialized before seeing the queries. The same materialization policy is then reused for all questions associated with that video.

The project compares several query-independent policies rather than learning a planner.

## Materialization types

- transcript
- sampled keyframes
- short summary

## Policy family

- Raw baseline
- Transcript-only
- Keyframes-only
- Summary-only
- Three-tier mixed policy:

Use a simple hand-written routing rule so the policy is runnable in experiments.

- transcript for speech-dense segments - keyframes for visually active segments - summary for low-information segments

## Token accounting

Keep build-time and query-time costs separate.

### Build-time token breakdown

- input video tokens
- input text tokens
- output transcript / summary tokens
- any other model-exposed generation tokens

### Query-time token breakdown

- input query text tokens
- input materialized text tokens
- input video tokens if the answering path still uses video
- output answer tokens
- output thinking tokens from Gemini

### Reported totals

For each policy, report:

- build-time totals
- query-time totals
- combined totals
- token breakdown by category
- workload accuracy

## Evaluation rule

Use deterministic benchmark scoring when the benchmark provides it. If not, use an LLM judge.

## Experiment plans

### Plan A: smallest credible result

- ActivityNet-QA only
- fixed-length segmentation
- all five policies
- run all questions for each video under one reused policy
- main figure: total tokens versus accuracy
- main table: per-policy token breakdown

### Plan B: stronger result

- repeat the same setup on Neptune
- compare whether the same policy ordering holds on a second dataset
- highlight whether long-video settings increase the value of segment-level mixed policies

### Plan C: token-shifting analysis

- test whether lower input-token policies cause higher output or thinking-token usage
- use Gemini thinking tokens directly
- report whether savings in one token bucket move cost into another bucket

## Minimal implementation checklist

- dataset loader grouped by video
- fixed-length segmenter
- transcript builder
- keyframe builder
- summary builder
- three-tier policy implementation
- token logger with build-time and query-time separation
- evaluator with deterministic scoring or LLM judge fallback

## Recommended execution order

1. Implement ActivityNet-QA first.
2. Freeze segmentation and the five policies.
3. Implement token logging before large runs.
4. Run a small pilot on a subset.
5. Produce the first token-versus-accuracy figure.
6. Add Neptune second if the first pipeline is stable.