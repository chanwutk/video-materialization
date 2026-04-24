"""GEPA-based optimization of the LLM routing policy directive."""

import asyncio
import hashlib
import json
import time

from google import genai

from .config import (
    GEPA_DIR,
    GEPA_MAX_METRIC_CALLS,
    MODEL_NAME,
    REFLECTION_MODEL_NAME,
    ROUTER_MODEL_NAME,
    SEGMENT_LENGTH_S,
)
from .evaluator import evaluate
from .policies import Policy, materialize_from_routing
from .router import SEED_DIRECTIVE, route_video_segments
from .runner import answer_question
from .segmenter import Segment, segment_video
from .tokens import TokenUsage


def _run_query_sync(
    client: genai.Client,
    video_id: str,
    youtube_url: str,
    segments: list[Segment],
    materialized: dict,
    entry: dict,
    model: str,
    video_duration_s: float | None,
    routing_hash: str | None = None,
) -> tuple[int | None, str, str, TokenUsage]:
    """Run answer_question synchronously via asyncio."""
    sem = asyncio.Semaphore(1)

    async def _run():
        return await answer_question(
            client, Policy.LLM_ROUTED, youtube_url, video_id, segments,
            materialized, entry, sem, model=model,
            video_duration_s=video_duration_s,
            routing_hash=routing_hash,
        )

    return asyncio.run(_run())


class GEPAEvaluatorState:
    """Holds shared state across GEPA evaluator calls."""

    def __init__(
        self,
        train_videos: dict[str, list[dict]],
        durations: dict[str, float],
        segment_length: int = SEGMENT_LENGTH_S,
        query_model: str = MODEL_NAME,
        router_model: str = ROUTER_MODEL_NAME,
    ):
        self.client = genai.Client()
        self.train_videos = train_videos
        self.durations = durations
        self.segment_length = segment_length
        self.query_model = query_model
        self.router_model = router_model

        # Pre-compute segments
        self.video_segments: dict[str, list[Segment]] = {}
        for vid in train_videos:
            self.video_segments[vid] = segment_video(
                durations[vid], segment_length,
            )

        # Build flat list of (video_id, entry) examples for GEPA
        self.examples: list[dict] = []
        for vid, entries in train_videos.items():
            for entry in entries:
                self.examples.append({
                    "video_id": vid,
                    "entry": entry,
                })

        # Cache routing decisions per (directive_hash, video_id) in memory
        self._routing_cache: dict[str, dict[str, list[str]]] = {}

    def _get_routing(self, directive: str, video_id: str) -> list[str]:
        """Get or compute routing decisions for a video with a given directive."""
        # route_video_segments already has disk caching; just call it
        return route_video_segments(
            self.client,
            video_id,
            self.video_segments[video_id],
            directive,
            model=self.router_model,
        )


def make_gepa_evaluator(state: GEPAEvaluatorState):
    """Create a GEPA-compatible evaluator function."""
    import gepa.optimize_anything as oa

    def evaluator(candidate: str, example: dict) -> tuple[float, dict]:
        video_id = example["video_id"]
        entry = example["entry"]
        segments = state.video_segments[video_id]
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

        # Get routing decisions
        routing_decisions = state._get_routing(candidate, video_id)
        r_hash = hashlib.sha256(candidate.encode()).hexdigest()[:16]

        # Materialize from routing
        materialized = materialize_from_routing(video_id, segments, routing_decisions)

        # Count tokens that will be used (text vs video)
        n_video_segments = sum(1 for m in materialized.values() if m.is_video)
        n_text_segments = sum(1 for m in materialized.values() if not m.is_video)
        n_skip = routing_decisions.count("SKIP")

        # Run the query
        predicted_id, raw_response, raw_thoughts, usage = _run_query_sync(
            state.client,
            video_id,
            youtube_url,
            segments,
            materialized,
            entry,
            state.query_model,
            state.durations.get(video_id),
            routing_hash=r_hash,
        )

        # Score
        correct = 1.0 if predicted_id == entry["answer_id"] else 0.0

        # Side info for GEPA reflection
        oa.log(f"Video: {video_id}, Question: {entry['question'][:100]}")
        oa.log(f"Correct: {correct}, Predicted: {predicted_id}, Actual: {entry['answer_id']}")
        oa.log(f"Routing: {n_video_segments} video, {n_text_segments} text, {n_skip} skipped")
        oa.log(f"Query tokens: {usage.total_tokens}")
        oa.log(f"Decisions: {routing_decisions}")

        side_info = {
            "query_tokens": usage.total_tokens,
            "n_video_segments": n_video_segments,
            "n_text_segments": n_text_segments,
            "n_skip": n_skip,
            "correct": correct,
        }
        return correct, side_info

    return evaluator


def run_gepa_optimization(
    train_videos: dict[str, list[dict]],
    durations: dict[str, float],
    segment_length: int = SEGMENT_LENGTH_S,
    query_model: str = MODEL_NAME,
    router_model: str = ROUTER_MODEL_NAME,
    max_metric_calls: int = GEPA_MAX_METRIC_CALLS,
) -> None:
    """Run GEPA to optimize the routing policy directive."""
    from gepa.optimize_anything import (
        EngineConfig,
        GEPAConfig,
        ReflectionConfig,
        optimize_anything,
    )

    print("\n=== GEPA Optimization ===")
    print(f"  Train videos: {len(train_videos)}")
    print(f"  Train questions: {sum(len(qs) for qs in train_videos.values())}")
    print(f"  Max metric calls: {max_metric_calls}")
    print(f"  Router model: {router_model}")
    print(f"  Query model: {query_model}")
    print(f"  Reflection model: {REFLECTION_MODEL_NAME}")

    state = GEPAEvaluatorState(
        train_videos=train_videos,
        durations=durations,
        segment_length=segment_length,
        query_model=query_model,
        router_model=router_model,
    )

    evaluator = make_gepa_evaluator(state)

    GEPA_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = str(GEPA_DIR / f"run_{int(time.time())}")

    config = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir,
            seed=42,
            display_progress_bar=True,
            max_metric_calls=max_metric_calls,
            candidate_selection_strategy="pareto",
            frontier_type="hybrid",
            parallel=False,  # We manage our own API concurrency
            cache_evaluation=True,
            cache_evaluation_storage="disk",
        ),
        reflection=ReflectionConfig(
            reflection_lm=REFLECTION_MODEL_NAME,
        ),
    )

    objective = (
        "Optimize the routing policy directive to maximize question-answering accuracy "
        "on video QA tasks. The directive guides an LLM router that decides, for each "
        "30-second video segment, whether to represent it as TRANSCRIPT (speech text), "
        "SUMMARY (text summary), LOW_FPS (low-frame-rate video), or SKIP (omit). "
        "Good directives produce routing decisions that preserve the information needed "
        "to answer questions correctly while using text representations (cheaper) where "
        "possible. Pay attention to patterns in the evaluation logs about which routing "
        "decisions lead to correct vs incorrect answers."
    )

    background = (
        "This is a video materialization system. Videos are divided into 30-second segments. "
        "Each segment can be represented as text (transcript or summary) or kept as video "
        "at low frame rate. Text is much cheaper in tokens but may lose visual information. "
        "The routing directive you're optimizing is a set of instructions that an LLM uses "
        "to decide the representation for each segment based on segment metadata "
        "(transcript text, summary text, word counts)."
    )

    print(f"\nStarting GEPA optimization...")
    print(f"  Seed directive:\n{SEED_DIRECTIVE}\n")

    result = optimize_anything(
        seed_candidate=SEED_DIRECTIVE,
        evaluator=evaluator,
        dataset=state.examples,
        objective=objective,
        background=background,
        config=config,
    )

    # Save results
    best_score = result.val_aggregate_scores[result.best_idx]
    best_candidate = result.best_candidate
    print(f"\n=== GEPA Optimization Complete ===")
    print(f"  Total candidates: {result.num_candidates}")
    print(f"  Total metric calls: {result.total_metric_calls}")
    print(f"  Best score: {best_score:.3f}")
    print(f"  Best directive:\n{best_candidate}")

    # Save all candidates and their scores for Pareto analysis
    all_candidates = []
    for i, (cand, score) in enumerate(zip(result.candidates, result.val_aggregate_scores)):
        directive_text = cand.get(result._str_candidate_key, str(cand)) if result._str_candidate_key else str(cand)
        all_candidates.append({
            "index": i,
            "score": score,
            "directive": directive_text,
        })

    output = {
        "best_score": best_score,
        "best_candidate": best_candidate,
        "all_candidates": all_candidates,
        "run_dir": run_dir,
    }
    output_path = GEPA_DIR / "optimization_result.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved optimization result to {output_path}")

    return result
