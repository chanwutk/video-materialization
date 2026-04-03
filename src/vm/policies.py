from enum import Enum

from google import genai

from .builders import build_transcript, build_keyframes, build_summary
from .config import SPEECH_DENSE_WORD_THRESHOLD, VISUALLY_ACTIVE_WORD_THRESHOLD, MODEL_NAME
from .segmenter import Segment
from .tokens import TokenUsage


class Policy(Enum):
    RAW = "raw"
    TRANSCRIPT = "transcript"
    KEYFRAMES = "keyframes"
    SUMMARY = "summary"
    MIXED = "mixed"


def _word_count(text: str) -> int:
    return len(text.split())


def _pick_mixed_material(
    transcript: str, keyframes: str, summary: str,
) -> tuple[str, str]:
    """Returns (chosen_text, chosen_type) for mixed policy routing."""
    if _word_count(transcript) > SPEECH_DENSE_WORD_THRESHOLD:
        return transcript, "transcript"
    if _word_count(keyframes) > VISUALLY_ACTIVE_WORD_THRESHOLD:
        return keyframes, "keyframes"
    return summary, "summary"


def materialize_video(
    client: genai.Client,
    policy: Policy,
    video_id: str,
    youtube_url: str,
    segments: list[Segment],
    model: str = MODEL_NAME,
) -> tuple[dict[int, str | None], list[TokenUsage]]:
    """
    Materialize all segments for a given policy.

    Returns:
        materialized: dict mapping segment index -> materialized text (None for RAW)
        build_usages: list of TokenUsage from builder calls
    """
    materialized: dict[int, str | None] = {}
    build_usages: list[TokenUsage] = []

    if policy == Policy.RAW:
        for seg in segments:
            materialized[seg.index] = None
        return materialized, build_usages

    for seg in segments:
        if policy == Policy.TRANSCRIPT:
            text, usage = build_transcript(client, video_id, youtube_url, seg, model=model)
            materialized[seg.index] = text
            build_usages.append(usage)

        elif policy == Policy.KEYFRAMES:
            text, usage = build_keyframes(client, video_id, youtube_url, seg, model=model)
            materialized[seg.index] = text
            build_usages.append(usage)

        elif policy == Policy.SUMMARY:
            text, usage = build_summary(client, video_id, youtube_url, seg, model=model)
            materialized[seg.index] = text
            build_usages.append(usage)

        elif policy == Policy.MIXED:
            # Build all three (cached if already done by other policies)
            t_text, t_usage = build_transcript(client, video_id, youtube_url, seg, model=model)
            k_text, k_usage = build_keyframes(client, video_id, youtube_url, seg, model=model)
            s_text, s_usage = build_summary(client, video_id, youtube_url, seg, model=model)
            build_usages.extend([t_usage, k_usage, s_usage])

            chosen_text, _ = _pick_mixed_material(t_text, k_text, s_text)
            materialized[seg.index] = chosen_text

    return materialized, build_usages
