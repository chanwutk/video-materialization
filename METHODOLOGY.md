# Methodology — GEPA-Optimized Video Materialization Routing

## 1. Problem Formulation

Given a video V divided into n fixed-length segments {s_1, s_2, ..., s_n} (each 30 seconds), we seek a **routing function** R(s_i) -> {TRANSCRIPT, SUMMARY, LOW_FPS, SKIP} that decides how to materialize each segment. The materialized representation replaces the raw video when answering downstream QA queries, trading off token cost against answer accuracy.

The **mixed** heuristic from the preliminary report used hard-coded word-count thresholds:

```
if transcript_words(s_i) > 30     -> TRANSCRIPT
elif transcript_words(s_i) > 0 
     OR summary_words(s_i) > 50   -> LOW_FPS
else                               -> SUMMARY
```

This is unprincipled — the thresholds are arbitrary and fixed regardless of content.

## 2. Our Approach: LLM Router with Evolved Policy Directive

We replace the heuristic with a two-stage approach:

**Stage 1 — LLM Router.** A lightweight LMM (gemini-2.5-flash-lite) receives a prompt with three sections:
1. **Fixed template**: describes the available materialization methods and output format
2. **Policy directive** (the part we optimize): natural language instructions for how to decide
3. **Segment metadata**: for each segment, the pre-computed transcript text (first 300 chars, word count) and summary text (first 300 chars, word count)

The router outputs a JSON array of n decisions, one per segment. The prompt uses Gemini's structured output (JSON schema) to guarantee valid responses.

**Stage 2 — Assembly.** Each routing decision maps to a pre-computed materialization:
- TRANSCRIPT: cached verbatim speech transcription
- SUMMARY: cached 2-3 sentence text summary
- LOW_FPS: original video at 0.2 FPS (kept as video tokens)
- SKIP: segment omitted entirely

The assembled mix of text and video parts is then passed to the QA model along with the question.

## 3. How GEPA Optimizes the Directive

The policy directive is a free-form natural language string (the seed is ~90 words based on the original heuristic). We use **GEPA** (Reflective Prompt Evolution) [1] to evolve it.

GEPA treats the directive as a **text parameter** and optimizes it through an evolutionary loop:

```
1. SEED: Initialize with the seed directive d_0 (our hand-written heuristic in natural language)

2. EVALUATE: Score d_0 on all training examples.
   For each (video, question) pair:
     a. Run the LLM router with d_0 on the video's segments -> routing decisions
     b. Assemble materialized content from cached transcripts/summaries + video
     c. Run QA model on assembled content + question -> predicted answer
     d. Score: 1.0 if correct, 0.0 if wrong
   Aggregate score = accuracy across all train examples

3. REFLECT: A reflection LMM (gemini-2.5-flash) receives:
     - The current directive d_i
     - Evaluation logs: which questions were correct/wrong, what routing decisions
       were made, token counts, video IDs
   The reflection LMM analyzes failure patterns and proposes an improved
   directive d_{i+1}

4. SELECT: GEPA maintains a Pareto frontier of candidates -- directives that each
   excel on different subsets of examples. When proposing the next candidate,
   GEPA selects a parent from this frontier (not just the single best).

5. REPEAT steps 2-4 until budget exhausted (max_metric_calls).
```

### Key properties of GEPA vs alternatives

**vs. random search**: GEPA's reflection step diagnoses *why* a directive failed (e.g., "visual segments were routed to TRANSCRIPT, losing critical visual context") rather than blindly sampling.

**vs. RL (e.g., GRPO)**: GEPA uses natural language reflection instead of scalar reward gradients — the reflection LMM reads full evaluation traces, not just a loss value. GEPA claims 35x fewer rollouts needed than RL approaches.

**Pareto-aware selection**: Rather than optimizing a single best directive, GEPA maintains a frontier of candidates. One directive might excel on dialogue-heavy videos, another on visual-action videos. This diversity in the population helps the evolutionary search escape local optima.

## 4. What the Evaluator Provides to GEPA

For each (candidate_directive, example) call, our evaluator returns:
- **Score**: 1.0 (correct) or 0.0 (incorrect)
- **Side information** (logged via oa.log(), fed to the reflection LMM):
  - Video ID and question text
  - Predicted vs actual answer
  - Routing breakdown: how many segments went to video / text / skip
  - Total query tokens consumed
  - The full routing decision array

This side information is what makes GEPA's reflection meaningful — the reflection LMM can see, for example, that a directive caused all segments of a cooking video to be routed to TRANSCRIPT, but the question asked "what color was the pot?" which requires visual information. It can then propose adding a rule about visual questions.

## 5. Experimental Setup

| Component | Model | Role |
|---|---|---|
| Materialization builders | gemini-3.1-pro-preview | One-time: generate transcripts + summaries per segment |
| LMM Router | gemini-2.5-flash-lite | Per-video: decide materialization per segment (cheapest model, text-only) |
| QA execution | gemini-2.5-flash | Per-question: answer questions against materialized content |
| GEPA reflection | gemini-2.5-flash | Per-iteration: analyze failures, propose improved directive |

- **Train set**: 10 videos, 100 questions (for GEPA optimization)
- **Test set**: 10 videos, 100 questions (held-out, for final evaluation)
- **GEPA budget**: 500 metric calls
- **Routing scope**: per-video (same routing for all questions on a video)

## 6. Narrative for the Report

The hand-tuned heuristic was a fixed function of word counts. We replaced it with a learned natural language policy, optimized by GEPA's reflective evolution loop that diagnoses routing failures and proposes improvements — a principled, data-driven approach. The seed directive encodes the same logic as the original heuristic but in natural language; GEPA then evolves it into something better by observing which routing decisions lead to correct vs incorrect answers.

## 7. GEPA Citation

```bibtex
@article{agrawal2025gepa,
  title={Reflective Prompt Evolution Can Outperform Reinforcement Learning},
  author={Agrawal, Lakshya A and Tan, Shangyin and Soylu, Dilara and Ziems, Noah and Khare, Rishi and Opsahl-Ong, Krista and Singhvi, Arnav and Shandilya, Herumb and Ryan, Michael J and Jiang, Meng and Potts, Christopher and Sen, Koushik and Dimakis, Alexandros G and Stoica, Ion and Klein, Dan and Zaharia, Matei and Khattab, Omar},
  journal={arXiv preprint arXiv:2507.19457},
  year={2025}
}
```
