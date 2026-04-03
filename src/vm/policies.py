import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

from google import genai

from .builders import (
    build_transcript, build_segment_summary,
    build_visual_description, build_whole_summary,
)
from .config import SPEECH_DENSE_WORD_THRESHOLD, VISUALLY_ACTIVE_WORD_THRESHOLD, MODEL_NAME
from .segmenter import Segment
from .tokens import TokenUsage


class Policy(Enum):
    RAW = "raw"
    TRANSCRIPT = "transcript"
    VISUAL_DESCRIPTION = "visual-description"
    SUMMARY = "summary"
    LOW_FPS = "low-fps"
    LOW_RES = "low-res"
    MIXED = "mixed"


@dataclass
class SegmentMaterial:
    text: str | None       # materialized text (None for video-based segments)
    material_type: str     # "transcript", "visual-description", "summary", "low-fps", "low-res", "raw"
    is_video: bool = False # True if query should send video Part instead of text


def _word_count(text: str) -> int:
    return len(text.split())


def _pick_mixed_material(
    transcript: str, summary: str,
) -> SegmentMaterial:
    """Route a segment for the mixed policy (3 tiers)."""
    if _word_count(transcript) > SPEECH_DENSE_WORD_THRESHOLD:
        return SegmentMaterial(text=transcript, material_type="transcript")
    if _word_count(transcript) > 0 or _word_count(summary) > VISUALLY_ACTIVE_WORD_THRESHOLD:
        return SegmentMaterial(text=None, material_type="low-res", is_video=True)
    return SegmentMaterial(text=summary, material_type="summary")


async def materialize_video(
    client: genai.Client,
    policy: Policy,
    video_id: str,
    youtube_url: str,
    segments: list[Segment],
    sem: asyncio.Semaphore,
    model: str = MODEL_NAME,
    pbar: Any = None,
) -> tuple[dict[int, SegmentMaterial], list[TokenUsage]]:
    """
    Materialize a video for a given policy. Fires builder calls concurrently.
    """
    materialized: dict[int, SegmentMaterial] = {}
    build_usages: list[TokenUsage] = []

    if policy == Policy.RAW:
        materialized[0] = SegmentMaterial(text=None, material_type="raw", is_video=True)
        return materialized, build_usages

    if policy == Policy.LOW_FPS:
        materialized[0] = SegmentMaterial(text=None, material_type="low-fps", is_video=True)
        return materialized, build_usages

    if policy == Policy.LOW_RES:
        materialized[0] = SegmentMaterial(text=None, material_type="low-res", is_video=True)
        return materialized, build_usages

    if policy == Policy.VISUAL_DESCRIPTION:
        text, usage = await build_visual_description(client, video_id, youtube_url, sem, model=model, pbar=pbar)
        build_usages.append(usage)
        materialized[0] = SegmentMaterial(text=text, material_type="visual-description")
        return materialized, build_usages

    if policy == Policy.SUMMARY:
        text, usage = await build_whole_summary(client, video_id, youtube_url, sem, model=model, pbar=pbar)
        build_usages.append(usage)
        materialized[0] = SegmentMaterial(text=text, material_type="summary")
        return materialized, build_usages

    if policy == Policy.TRANSCRIPT:
        tasks = [
            build_transcript(client, video_id, youtube_url, seg, sem, model=model, pbar=pbar)
            for seg in segments
        ]
        results = await asyncio.gather(*tasks)
        for seg, (text, usage) in zip(segments, results):
            build_usages.append(usage)
            materialized[seg.index] = SegmentMaterial(text=text, material_type="transcript")
        return materialized, build_usages

    if policy == Policy.MIXED:
        transcript_tasks = [
            build_transcript(client, video_id, youtube_url, seg, sem, model=model, pbar=pbar)
            for seg in segments
        ]
        summary_tasks = [
            build_segment_summary(client, video_id, youtube_url, seg, sem, model=model, pbar=pbar)
            for seg in segments
        ]
        all_results = await asyncio.gather(*transcript_tasks, *summary_tasks)
        n = len(segments)
        transcript_results = all_results[:n]
        summary_results = all_results[n:]

        for seg, (t_text, t_usage), (s_text, s_usage) in zip(segments, transcript_results, summary_results):
            build_usages.extend([t_usage, s_usage])
            materialized[seg.index] = _pick_mixed_material(t_text, s_text)
        return materialized, build_usages

    raise ValueError(f"Unknown policy: {policy}")
