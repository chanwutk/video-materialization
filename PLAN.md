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

## Prompt construction

### Build phase (materialization)

Each policy pre-computes materials by calling Gemini with a segment or whole-video Part plus a task prompt. RAW, LOW_FPS, and LOW_RES skip the build phase entirely.

| Policy | Scope | Build prompt |
|---|---|---|
| `transcript` | per segment | "Transcribe all speech in this video segment verbatim. If there is no speech, respond with [NO SPEECH]." |
| `visual-description` | whole video | "Describe the key visual scenes in this video in chronological order. For each distinct scene, provide a one-sentence description of what is shown." |
| `summary` | whole video | "Provide a concise summary of this video covering both visual content and any speech." |
| `mixed` | per segment | Both transcript and per-segment summary prompts run concurrently; routing decides which to use. Per-segment summary prompt: "Provide a concise summary of this video segment in 2-3 sentences, covering both visual content and any speech." |

**Mixed routing rule** (thresholds in `config.py`):
1. transcript word count > 30 → use transcript text
2. transcript word count > 0 OR summary word count > 50 → use low-res video clip
3. otherwise → use summary text

### Query phase (prompt construction)

All policies append the same question block:
```
You will be given a question about a video and five possible answer options.
Question: <question>
Possible answer choices:
1) <choice 0>  2) <choice 1>  3) <choice 2>  4) <choice 3>  5) <choice 4>

Output the final answer in the format "Final Answer: (X)" where X is the correct digit choice.
Do not output any explanation or the full answer text.
```

**RAW:** `[whole video Part] + [question text]`

**LOW_FPS:** `[whole video Part @ 0.2 fps] + [question text]`

**LOW_RES:** `[whole video Part @ MEDIA_RESOLUTION_LOW] + [question text]`

**VISUAL_DESCRIPTION / SUMMARY** (text-only, single Part):
```
[MATERIALIZATION_PREAMBLE]
[visual-description OR summary]: <built text>

[question text]
```

**TRANSCRIPT** (text-only, segmented):
```
[MATERIALIZATION_PREAMBLE]
Segment 0 (0s-30s) [transcript]: <text>
Segment 1 (30s-60s) [transcript]: <text>
...

[question text]
```

**MIXED** (multimodal Parts list — text or video clip per segment):
```
[MATERIALIZATION_PREAMBLE text Part]
Segment 0 (0s-30s) [transcript]: <text>          ← text Part
Segment 1 (30s-60s) [low-res]:                   ← label text Part + low-res video clip Part
Segment 2 (60s-90s) [summary]: <text>            ← text Part
...
[question text Part]
```

The `MATERIALIZATION_PREAMBLE` (defined in `runner.py`) explains each material type label to the model so it interprets each piece of context correctly.

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