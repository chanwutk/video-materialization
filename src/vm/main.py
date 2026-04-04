import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import altair as alt
import pandas as pd
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from .config import (
    MODEL_NAME, SEGMENT_LENGTH_S, TOP_K_VIDEOS,
    RESULTS_DIR, DATA_DIR,
)
from .dataset import download_minerva, group_by_video, select_top_k
from .duration import get_durations_for_videos
from .evaluator import evaluate
from .policies import (
    PHASE1_PREBUILD_POLICIES,
    Policy,
    materialize_video,
    phase1_prebuild_total_calls,
    prebuild_gemini_call_count,
)
from .runner import answer_question
from .segmenter import segment_video
from .tokens import PolicyTokenLog, TokenUsage

ALL_POLICIES = "raw,transcript,visual-description,summary,low-fps,mixed"
DEFAULT_CONCURRENCY = 16
MAX_VIDEO_DURATION_S = 1500

# Description, bar, completed/total, %, ETA.
_PROGRESS_COLUMNS = (
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    # TimeRemainingColumn(),
)


@dataclass
class Pbar:
    """Progress bar adapter for asyncio.Semaphore."""

    progress: Progress
    task_id: int

    def update(self, n: int = 1) -> None:
        self.progress.update(self.task_id, advance=n)


def _latency_stats(usages: list[TokenUsage]) -> tuple[float, int, float]:
    """Return summed latency, live-call count, and mean latency over live calls."""
    live_latencies = [u.latency_s for u in usages if u.latency_s > 0]
    if not live_latencies:
        return 0.0, 0, 0.0

    total_s = float(sum(live_latencies))
    n_live = len(live_latencies)
    return total_s, n_live, total_s / n_live


def save_tokens_vs_accuracy_chart(
    plot_df: pd.DataFrame,
    x_field: str,
    x_title: str,
    out_path: Path,
    chart_title: str,
) -> None:
    """Scatter of token cost vs accuracy; writes PNG via Altair/Vega."""
    points = (
        alt.Chart(plot_df)
        .mark_circle(size=120, opacity=0.95)
        .encode(
            x=alt.X(f"{x_field}:Q", title=x_title),
            y=alt.Y("accuracy_pct:Q", title="Accuracy (%)"),
            tooltip=[
                alt.Tooltip("policy:N", title="Policy"),
                alt.Tooltip(f"{x_field}:Q", format=",", title=x_title),
                alt.Tooltip("accuracy_pct:Q", format=".2f", title="Accuracy (%)"),
            ],
        )
    )
    labels = (
        alt.Chart(plot_df)
        .mark_text(dx=10, dy=5, fontSize=11, align="left", baseline="middle")
        .encode(
            x=f"{x_field}:Q",
            y="accuracy_pct:Q",
            text="policy:N",
        )
    )
    chart = (
        (points + labels)
        .properties(
            title=chart_title,
            width=520,
            height=400,
        )
        .configure_axis(grid=True, gridOpacity=0.3)
        .configure_view(strokeWidth=0)
    )
    chart.save(str(out_path), scale_factor=2)


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
    """Download dataset, group, filter by duration, then select top-K among eligible."""
    entries = download_minerva()
    grouped = group_by_video(entries)

    all_ids = list(grouped.keys())
    print(f"\nFetching durations for {len(all_ids)} videos...")
    durations = get_durations_for_videos(all_ids)

    eligible = {
        vid: grouped[vid]
        for vid in grouped
        if vid in durations and durations[vid] <= MAX_VIDEO_DURATION_S
    }
    print(
        f"  {len(eligible)} videos with known duration <= {MAX_VIDEO_DURATION_S}s "
        f"(excluded {len(grouped) - len(eligible)} longer or unavailable)",
    )

    selected = select_top_k(eligible, top_k, duration_hint=durations)

    print(f"\nFinal selection: {len(selected)} videos")
    for vid, qs in selected.items():
        print(f"  {vid}: {len(qs)} questions, {durations[vid]:.0f}s duration")

    return selected, durations


async def run_experiment_async(args):
    selected, durations = select_videos(args.top_k)
    # Longest videos first for build and query execution order.
    ordered_vids = sorted(
        selected.keys(), key=lambda v: durations[v], reverse=True,
    )

    policies = [Policy(p.strip()) for p in args.policies.split(",")]
    segment_length = args.segment_length
    model = args.model

    # Compute segments for each video
    video_segments = {}
    for vid in ordered_vids:
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
        for vid in ordered_vids:
            segs = video_segments[vid]
            print(f"  {vid}: {len(segs)} segments, {len(selected[vid])} questions")

        n_builder_calls = phase1_prebuild_total_calls(video_segments)
        n_query_calls = total_questions * len(policies)

        print(f"\nEstimated API calls:")
        print(f"  Builder calls: {n_builder_calls}")
        for p in PHASE1_PREBUILD_POLICIES:
            per_run = sum(prebuild_gemini_call_count(p, len(segs)) for segs in video_segments.values())
            print(f"    {p.value}: {per_run}")
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
    n_build_calls = phase1_prebuild_total_calls(video_segments)

    build_tasks = []
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task_id = progress.add_task("  Building", total=n_build_calls)
        pbar = Pbar(progress, task_id)
        for vid in ordered_vids:
            youtube_url = f"https://www.youtube.com/watch?v={vid}"
            segs = video_segments[vid]
            for policy in PHASE1_PREBUILD_POLICIES:
                build_tasks.append(
                    materialize_video(
                        client, policy, vid, youtube_url, segs, sem, model=model, pbar=pbar,
                    ),
                )

        await asyncio.gather(*build_tasks)
    print("  Build phase complete.")

    # Phase 2: Load materializations and run all policy×question queries in parallel
    # (ordering is still bounded by --concurrency via the shared semaphore).
    print("\n=== Phase 2: Running queries ===")
    all_results: dict[str, list[dict]] = {p.value: [] for p in policies}
    all_token_logs: dict[str, list[PolicyTokenLog]] = {p.value: [] for p in policies}

    total_questions = sum(len(qs) for qs in selected.values())
    total_query_tasks = total_questions * len(policies)

    mat_keys = [(p, vid) for p in policies for vid in ordered_vids]
    materialize_tasks = [
        materialize_video(
            client,
            p,
            vid,
            f"https://www.youtube.com/watch?v={vid}",
            video_segments[vid],
            sem,
            model=model,
        )
        for p, vid in mat_keys
    ]
    materialize_results = await asyncio.gather(*materialize_tasks)
    video_materials = {
        (p.value, vid): res for (p, vid), res in zip(mat_keys, materialize_results)
    }

    token_log_by_pv: dict[tuple[str, str], PolicyTokenLog] = {}
    for policy in policies:
        for vid in ordered_vids:
            materialized, build_usages = video_materials[(policy.value, vid)]
            token_log = PolicyTokenLog(
                policy=policy.value,
                video_id=vid,
                build_usage=build_usages,
            )
            all_token_logs[policy.value].append(token_log)
            token_log_by_pv[(policy.value, vid)] = token_log

    query_tasks: list = []
    task_meta: list = []
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task_id = progress.add_task("  Queries", total=total_query_tasks)
        pbar = Pbar(progress, task_id)
        for policy in policies:
            for vid in ordered_vids:
                youtube_url = f"https://www.youtube.com/watch?v={vid}"
                segs = video_segments[vid]
                materialized, _ = video_materials[(policy.value, vid)]
                token_log = token_log_by_pv[(policy.value, vid)]
                for entry in selected[vid]:
                    query_tasks.append(
                        answer_question(
                            client, policy, youtube_url, vid, segs, materialized,
                            entry, sem, model=model, pbar=pbar,
                            video_duration_s=durations[vid],
                        ),
                    )
                    task_meta.append((policy, vid, entry, token_log))

        query_results = await asyncio.gather(*query_tasks)

    for (policy, vid, entry, token_log), (
        predicted_id, raw_response, raw_thoughts, usage,
    ) in zip(task_meta, query_results):
        token_log.query_usage.append(usage)
        all_results[policy.value].append({
            "video_id": vid,
            "question_key": entry["key"],
            "predicted_id": predicted_id,
            "answer_id": entry["answer_id"],
            "policy": policy.value,
            "raw_response": raw_response,
            "raw_thoughts": raw_thoughts,
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

        build_usages_flat: list[TokenUsage] = []
        query_usages_flat: list[TokenUsage] = []
        for log in logs:
            build_usages_flat.extend(log.build_usage)
            query_usages_flat.extend(log.query_usage)
        build_lat_s, build_n_api, build_mean_s = _latency_stats(build_usages_flat)
        query_lat_s, query_n_api, query_mean_s = _latency_stats(query_usages_flat)

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
            "build_api_latency_s_sum": build_lat_s,
            "build_api_calls_live": build_n_api,
            "build_api_latency_mean_s": build_mean_s,
            "query_api_latency_s_sum": query_lat_s,
            "query_api_calls_live": query_n_api,
            "query_api_latency_mean_s": query_mean_s,
        }
        summary_rows.append(row)

        print(f"\n{policy.value}:")
        print(f"  Accuracy: {eval_result['accuracy']:.1%} ({eval_result['correct']}/{eval_result['total']})")
        print(f"  Unparsed: {eval_result['unparsed']}")
        print(f"  Build tokens: {total_build:,}")
        print(f"  Query tokens: {total_query:,}")
        print(f"  Total tokens: {total:,}")
        print(
            f"  Build API latency: {build_lat_s:.1f}s sum over {build_n_api} calls "
            f"(mean {build_mean_s:.2f}s)"
        )
        print(
            f"  Query API latency: {query_lat_s:.1f}s sum over {query_n_api} calls "
            f"(mean {query_mean_s:.2f}s)"
        )

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

    # Plot tokens vs accuracy (pandas + Altair; no Python loops on rows)
    plot_df = df.assign(accuracy_pct=df["accuracy"].mul(100))
    chart_specs = [
        (
            "total_tokens",
            "Total Tokens (build + query)",
            RESULTS_DIR / "tokens_vs_accuracy.png",
            "Token Cost vs Accuracy by Policy",
        ),
        (
            "build_total_tokens",
            "Build Tokens",
            RESULTS_DIR / "tokens_vs_accuracy_build.png",
            "Token Cost vs Accuracy by Policy (build only)",
        ),
        (
            "query_total_tokens",
            "Query Tokens",
            RESULTS_DIR / "tokens_vs_accuracy_query.png",
            "Token Cost vs Accuracy by Policy (query only)",
        ),
    ]
    for x_field, x_title, png_path, chart_title in chart_specs:
        save_tokens_vs_accuracy_chart(plot_df, x_field, x_title, png_path, chart_title)
        print(f"Saved {png_path.name}")


def main():
    args = parse_args()
    asyncio.run(run_experiment_async(args))


if __name__ == "__main__":
    main()
