import argparse
import asyncio
import json

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from .config import (
    MODEL_NAME, SEGMENT_LENGTH_S, TOP_K_VIDEOS, TOP_K_CANDIDATES,
    RESULTS_DIR, DATA_DIR,
)
from .dataset import download_minerva, group_by_video, select_top_k
from .duration import get_durations_for_videos
from .evaluator import evaluate
from .policies import Policy, materialize_video
from .runner import answer_question
from .segmenter import segment_video
from .tokens import PolicyTokenLog

ALL_POLICIES = "raw,transcript,visual-description,summary,low-fps,low-res,mixed"
DEFAULT_CONCURRENCY = 128


def parse_args():
    parser = argparse.ArgumentParser(description="Video Materialization Experiment")
    parser.add_argument("--top-k", type=int, default=TOP_K_VIDEOS)
    parser.add_argument("--segment-length", type=int, default=SEGMENT_LENGTH_S)
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--policies", type=str, default=ALL_POLICIES,
        help="Comma-separated list of policies to run",
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Max concurrent Gemini API calls")
    parser.add_argument("--dry-run", action="store_true", help="Load data and print plan without API calls")
    return parser.parse_args()


def select_videos(top_k: int):
    """Download dataset, group, select top-K with valid durations."""
    entries = download_minerva()
    grouped = group_by_video(entries)

    candidates = select_top_k(grouped, TOP_K_CANDIDATES)

    print(f"\nFetching durations for {len(candidates)} candidate videos...")
    durations = get_durations_for_videos(list(candidates.keys()))

    valid_videos = {
        vid: candidates[vid]
        for vid in candidates
        if vid in durations
    }
    sorted_valid = sorted(valid_videos.items(), key=lambda x: len(x[1]), reverse=True)
    selected = dict(sorted_valid[:top_k])

    print(f"\nFinal selection: {len(selected)} videos with valid durations")
    for vid, qs in selected.items():
        print(f"  {vid}: {len(qs)} questions, {durations[vid]:.0f}s duration")

    return selected, durations


async def run_experiment_async(args):
    selected, durations = select_videos(args.top_k)

    policies = [Policy(p.strip()) for p in args.policies.split(",")]
    segment_length = args.segment_length
    model = args.model

    # Compute segments for each video
    video_segments = {}
    for vid in selected:
        segs = segment_video(durations[vid], segment_length)
        video_segments[vid] = segs

    if args.dry_run:
        print(f"\n=== DRY RUN ===")
        print(f"Model: {model}")
        print(f"Segment length: {segment_length}s")
        print(f"Policies: {[p.value for p in policies]}")
        print(f"Concurrency: {args.concurrency}")
        print(f"Videos: {len(selected)}")
        total_segments = sum(len(s) for s in video_segments.values())
        total_questions = sum(len(qs) for qs in selected.values())
        print(f"Total segments: {total_segments}")
        print(f"Total questions: {total_questions}")
        print(f"\nPer-video breakdown:")
        for vid, segs in video_segments.items():
            print(f"  {vid}: {len(segs)} segments, {len(selected[vid])} questions")

        n_videos = len(selected)
        n_transcript_calls = total_segments
        n_seg_summary_calls = total_segments
        n_whole_video_calls = n_videos * 2
        n_builder_calls = n_transcript_calls + n_seg_summary_calls + n_whole_video_calls
        n_query_calls = total_questions * len(policies)

        print(f"\nEstimated API calls:")
        print(f"  Builder calls: {n_builder_calls}")
        print(f"    Transcript (per-segment): {n_transcript_calls}")
        print(f"    Segment summary (per-segment, for mixed): {n_seg_summary_calls}")
        print(f"    Whole-video (visual-desc + summary): {n_whole_video_calls}")
        print(f"  Query calls: {n_query_calls}")
        n_video_query = sum(
            total_questions for p in policies
            if p in (Policy.RAW, Policy.LOW_FPS, Policy.LOW_RES)
        )
        print(f"    (of which video-input queries: {n_video_query})")
        return

    # Initialize Gemini client and semaphore
    from google import genai
    client = genai.Client()
    sem = asyncio.Semaphore(args.concurrency)

    # Phase 1: Pre-build all materializations concurrently
    print("\n=== Phase 1: Building materializations ===")
    # Count individual builder calls for accurate progress
    total_segments = sum(len(s) for s in video_segments.values())
    n_videos = len(selected)
    # transcript (per-seg) + summary (per-seg, for mixed, transcript cached) + 2 whole-video
    n_build_calls = total_segments + total_segments + n_videos * 2

    build_tasks = []
    with tqdm(total=n_build_calls, desc="  Building") as pbar:
        for vid in selected:
            youtube_url = f"https://www.youtube.com/watch?v={vid}"
            segs = video_segments[vid]
            build_tasks.append(materialize_video(client, Policy.TRANSCRIPT, vid, youtube_url, segs, sem, model=model, pbar=pbar))
            build_tasks.append(materialize_video(client, Policy.MIXED, vid, youtube_url, segs, sem, model=model, pbar=pbar))
            build_tasks.append(materialize_video(client, Policy.VISUAL_DESCRIPTION, vid, youtube_url, segs, sem, model=model, pbar=pbar))
            build_tasks.append(materialize_video(client, Policy.SUMMARY, vid, youtube_url, segs, sem, model=model, pbar=pbar))

        await asyncio.gather(*build_tasks)
    print("  Build phase complete.")

    # Phase 2: Run queries for each policy concurrently
    print("\n=== Phase 2: Running queries ===")
    all_results: dict[str, list[dict]] = {p.value: [] for p in policies}
    all_token_logs: dict[str, list[PolicyTokenLog]] = {p.value: [] for p in policies}

    total_questions = sum(len(qs) for qs in selected.values())

    for policy in policies:
        # Pre-compute materialized data for all videos (all cached from Phase 1)
        video_materials: dict[str, tuple] = {}  # vid -> (materialized, build_usages)
        for vid in selected:
            youtube_url = f"https://www.youtube.com/watch?v={vid}"
            segs = video_segments[vid]
            materialized, build_usages = await materialize_video(
                client, policy, vid, youtube_url, segs, sem, model=model,
            )
            video_materials[vid] = (materialized, build_usages)

        # Build all query tasks
        query_tasks = []
        task_meta = []
        with tqdm(total=total_questions, desc=f"  {policy.value}") as pbar:
            for vid in selected:
                youtube_url = f"https://www.youtube.com/watch?v={vid}"
                segs = video_segments[vid]
                materialized, build_usages = video_materials[vid]

                token_log = PolicyTokenLog(
                    policy=policy.value,
                    video_id=vid,
                    build_usage=build_usages,
                )
                all_token_logs[policy.value].append(token_log)

                for entry in selected[vid]:
                    task = answer_question(
                        client, policy, youtube_url, vid, segs, materialized,
                        entry, sem, model=model, pbar=pbar,
                    )
                    query_tasks.append(task)
                    task_meta.append((vid, entry, token_log))

            query_results = await asyncio.gather(*query_tasks)

        for (vid, entry, token_log), (predicted_id, raw_response, usage) in zip(task_meta, query_results):
            token_log.query_usage.append(usage)
            all_results[policy.value].append({
                "video_id": vid,
                "question_key": entry["key"],
                "predicted_id": predicted_id,
                "answer_id": entry["answer_id"],
                "policy": policy.value,
                "raw_response": raw_response,
            })

    # Phase 3: Evaluate and output
    print("\n=== Phase 3: Evaluation ===")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for policy in policies:
        preds = all_results[policy.value]
        eval_result = evaluate(preds)
        logs = all_token_logs[policy.value]

        total_build = sum(log.total_build_tokens for log in logs)
        total_query = sum(log.total_query_tokens for log in logs)
        total = total_build + total_query

        build_prompt = sum(sum(u.prompt_tokens for u in log.build_usage) for log in logs)
        build_output = sum(sum(u.candidates_tokens for u in log.build_usage) for log in logs)
        build_thoughts = sum(sum(u.thoughts_tokens for u in log.build_usage) for log in logs)
        query_prompt = sum(sum(u.prompt_tokens for u in log.query_usage) for log in logs)
        query_output = sum(sum(u.candidates_tokens for u in log.query_usage) for log in logs)
        query_thoughts = sum(sum(u.thoughts_tokens for u in log.query_usage) for log in logs)

        row = {
            "policy": policy.value,
            "accuracy": eval_result["accuracy"],
            "correct": eval_result["correct"],
            "total": eval_result["total"],
            "unparsed": eval_result["unparsed"],
            "build_prompt_tokens": build_prompt,
            "build_output_tokens": build_output,
            "build_thoughts_tokens": build_thoughts,
            "build_total_tokens": total_build,
            "query_prompt_tokens": query_prompt,
            "query_output_tokens": query_output,
            "query_thoughts_tokens": query_thoughts,
            "query_total_tokens": total_query,
            "total_tokens": total,
        }
        summary_rows.append(row)

        print(f"\n{policy.value}:")
        print(f"  Accuracy: {eval_result['accuracy']:.1%} ({eval_result['correct']}/{eval_result['total']})")
        print(f"  Unparsed: {eval_result['unparsed']}")
        print(f"  Build tokens: {total_build:,}")
        print(f"  Query tokens: {total_query:,}")
        print(f"  Total tokens: {total:,}")

    # Save token breakdown table
    df = pd.DataFrame(summary_rows)
    df.to_csv(RESULTS_DIR / "token_breakdown.csv", index=False)
    print(f"\nSaved token_breakdown.csv")

    # Save raw results
    raw_output = {
        "results": {p: all_results[p] for p in all_results},
        "token_logs": {
            p: [log.to_dict() for log in all_token_logs[p]]
            for p in all_token_logs
        },
    }
    (RESULTS_DIR / "raw_results.json").write_text(json.dumps(raw_output, indent=2))
    print("Saved raw_results.json")

    # Plot tokens vs accuracy
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in summary_rows:
        ax.scatter(row["total_tokens"], row["accuracy"] * 100, s=100, zorder=5)
        ax.annotate(
            row["policy"],
            (row["total_tokens"], row["accuracy"] * 100),
            textcoords="offset points", xytext=(10, 5), fontsize=11,
        )
    ax.set_xlabel("Total Tokens (build + query)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Token Cost vs Accuracy by Policy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "tokens_vs_accuracy.png", dpi=150)
    print("Saved tokens_vs_accuracy.png")


def main():
    args = parse_args()
    asyncio.run(run_experiment_async(args))


if __name__ == "__main__":
    main()
