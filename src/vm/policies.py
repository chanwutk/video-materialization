from dataclasses import dataclass
from enum import Enum

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
    MIXED = "mixed"


@dataclass
class SegmentMaterial:
    text: str | None       # materialized text (None for video-based segments)
    material_type: str     # "transcript", "visual-description", "summary", "low-fps", "raw"
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
        # Has some content suggesting visual activity -> low-fps video
        return SegmentMaterial(text=None, material_type="low-fps", is_video=True)
    return SegmentMaterial(text=summary, material_type="summary")


def materialize_video(
    client: genai.Client,
    policy: Policy,
    video_id: str,
    youtube_url: str,
    segments: list[Segment],
    model: str = MODEL_NAME,
) -> tuple[dict[int, SegmentMaterial], list[TokenUsage]]:
    """
    Materialize a video for a given policy.

    Returns:
        materialized: dict mapping segment index -> SegmentMaterial
        build_usages: list of TokenUsage from builder calls
    """
    materialized: dict[int, SegmentMaterial] = {}
    build_usages: list[TokenUsage] = []

    if policy == Policy.RAW:
        # No build, query sends full video at default fps
        materialized[0] = SegmentMaterial(text=None, material_type="raw", is_video=True)
        return materialized, build_usages

    if policy == Policy.LOW_FPS:
        # No build, query sends full video at low fps
        materialized[0] = SegmentMaterial(text=None, material_type="low-fps", is_video=True)
        return materialized, build_usages

    if policy == Policy.VISUAL_DESCRIPTION:
        # Whole-video visual description
        text, usage = build_visual_description(client, video_id, youtube_url, model=model)
        build_usages.append(usage)
        materialized[0] = SegmentMaterial(text=text, material_type="visual-description")
        return materialized, build_usages

    if policy == Policy.SUMMARY:
        # Whole-video summary
        text, usage = build_whole_summary(client, video_id, youtube_url, model=model)
        build_usages.append(usage)
        materialized[0] = SegmentMaterial(text=text, material_type="summary")
        return materialized, build_usages

    if policy == Policy.TRANSCRIPT:
        # Per-segment transcript
        for seg in segments:
            text, usage = build_transcript(client, video_id, youtube_url, seg, model=model)
            build_usages.append(usage)
            materialized[seg.index] = SegmentMaterial(text=text, material_type="transcript")
        return materialized, build_usages

    if policy == Policy.MIXED:
        # Per-segment: build transcript + summary, then route
        for seg in segments:
            t_text, t_usage = build_transcript(client, video_id, youtube_url, seg, model=model)
            s_text, s_usage = build_segment_summary(client, video_id, youtube_url, seg, model=model)
            build_usages.extend([t_usage, s_usage])
            materialized[seg.index] = _pick_mixed_material(t_text, s_text)
        return materialized, build_usages

    raise ValueError(f"Unknown policy: {policy}")
