import asyncio
import json
import re
import time
from typing import Any

from google import genai
from google.genai import types

from .cache import (
    answer_cache_key,
    cache_key,
    load_answer_cache,
    load_builder_cache,
    save_answer_cache,
)
from .config import (
    LOW_FPS_RATE,
    LOW_RES_MEDIA_RESOLUTION,
    MODEL_NAME,
    VIDEO_DURATION_CLIP_SLACK_S,
)


def _effective_low_fps(span_s: float) -> float:
    """
    Gemini rejects low-fps clips when the window is too short for the requested
    fps (e.g. 0.2 fps needs ~5s for one frame). Bump fps so span * fps >= 1.
    """
    if span_s <= 0:
        return 1.0
    if span_s * LOW_FPS_RATE >= 1.0:
        return LOW_FPS_RATE
    return max(LOW_FPS_RATE, 1.0 / span_s)
from .genai_config import get_qa_config
from .retry import with_retries_async
from .genai_response import split_main_and_thought_texts
from .policies import Policy, SegmentMaterial
from .segmenter import Segment
from .tokens import TokenUsage


LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
DIGIT_TO_INDEX = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
ANSWER_PROMPT_CACHE_VERSION = "minerva-json-schema-v2"

MATERIALIZATION_PREAMBLE = """\
Below is pre-computed context for a video.
Each piece of context is labeled with its materialization type:
- [transcript]: Verbatim speech transcription from the audio track.
- [visual-description]: Descriptions of key visual scenes and actions.
- [summary]: A brief combined overview of both visual and audio content.
- [low-fps]: A low-frame-rate video clip of the segment (visual only).
- [low-res]: A low-resolution video clip of the segment.

Use the type labels to interpret each piece of context appropriately.
"""


def _build_question_text(entry: dict) -> str:
    q = entry["question"]
    choices = [
        f"1) {entry['answer_choice_0']}",
        f"2) {entry['answer_choice_1']}",
        f"3) {entry['answer_choice_2']}",
        f"4) {entry['answer_choice_3']}",
        f"5) {entry['answer_choice_4']}",
    ]
    lines = [
        "You will be given a question about a video and five possible answer options.",
        f"Question: {q}",
        "Possible answer choices:",
        *choices,
        "",
        'Respond with JSON only: an object with integer field "choice" equal to '
        "the correct option number (1, 2, 3, 4, or 5).",
    ]
    return "\n".join(lines)


def _build_nonsegmented_text_prompt(
    material: SegmentMaterial,
    entry: dict,
) -> str:
    return (
        MATERIALIZATION_PREAMBLE + "\n"
        f"[{material.material_type}]: {material.text}\n\n"
        + _build_question_text(entry)
    )


def _build_segmented_text_prompt(
    materialized: dict[int, SegmentMaterial],
    segments: list[Segment],
    entry: dict,
) -> str:
    lines = [MATERIALIZATION_PREAMBLE]
    for seg in segments:
        mat = materialized[seg.index]
        lines.append(f"Segment {seg.index} ({seg.start_s}s-{seg.end_s}s) [{mat.material_type}]: {mat.text}")
    lines.append("")
    lines.append(_build_question_text(entry))
    return "\n".join(lines)


def _mixed_clip_bounds(
    seg: Segment, video_duration_s: float | None,
) -> tuple[int, int] | None:
    """
    Shrink [start,end) so end does not exceed a conservative stream length.
    Cached duration often rounds up vs what YouTube/Gemini can decode →
    INVALID_ARGUMENT / no frames when the window is past the real end.
    """
    start, end = seg.start_s, seg.end_s
    if end <= start:
        return None
    if video_duration_s is None:
        return start, end
    safe_upper = int(video_duration_s) - VIDEO_DURATION_CLIP_SLACK_S
    clip_end = min(end, safe_upper)
    if clip_end <= start:
        return None
    return start, clip_end


def _mixed_segment_fallback_text(video_id: str, segment_index: int) -> str:
    for kind in ("summary", "transcript"):
        cached = load_builder_cache(cache_key(video_id, segment_index, kind))
        if cached and cached.get("text"):
            return str(cached["text"])
    return "[NO SEGMENT CONTEXT]"


def _build_mixed_parts(
    materialized: dict[int, SegmentMaterial],
    segments: list[Segment],
    youtube_url: str,
    entry: dict,
    video_id: str,
    video_duration_s: float | None,
) -> list[types.Part]:
    parts = [types.Part(text=MATERIALIZATION_PREAMBLE)]
    for seg in segments:
        mat = materialized[seg.index]
        if mat.is_video:
            if mat.material_type == "low-res":
                parts.append(types.Part(
                    text=f"Segment {seg.index} ({seg.start_s}s-{seg.end_s}s) [{mat.material_type}]:",
                ))
                parts.append(types.Part(
                    file_data=types.FileData(file_uri=youtube_url),
                    video_metadata=types.VideoMetadata(
                        start_offset=f"{seg.start_s}s",
                        end_offset=f"{seg.end_s}s",
                    ),
                    media_resolution=types.PartMediaResolution(level=LOW_RES_MEDIA_RESOLUTION),
                ))
            elif mat.material_type == "low-fps":
                bounds = _mixed_clip_bounds(seg, video_duration_s)
                if bounds is None:
                    fb = _mixed_segment_fallback_text(video_id, seg.index)
                    parts.append(types.Part(
                        text=f"Segment {seg.index} ({seg.start_s}s-{seg.end_s}s) "
                        f"[{mat.material_type}]: {fb}"
                    ))
                else:
                    c_start, c_end = bounds
                    dur = float(c_end - c_start)
                    label = (
                        f"Segment {seg.index} ({c_start}s-{c_end}s) [{mat.material_type}]"
                    )
                    if (c_start, c_end) != (seg.start_s, seg.end_s):
                        label += f" (clipped from {seg.start_s}s-{seg.end_s}s)"
                    parts.append(types.Part(text=f"{label}:"))
                    parts.append(types.Part(
                        file_data=types.FileData(file_uri=youtube_url),
                        video_metadata=types.VideoMetadata(
                            start_offset=f"{c_start}s",
                            end_offset=f"{c_end}s",
                            fps=_effective_low_fps(dur),
                        ),
                    ))
            else:
                raise ValueError(f"Unknown mixed video material_type: {mat.material_type}")
        else:
            parts.append(types.Part(
                text=f"Segment {seg.index} ({seg.start_s}s-{seg.end_s}s) [{mat.material_type}]: {mat.text}"
            ))
    parts.append(types.Part(text=_build_question_text(entry)))
    return parts


def _build_video_parts(
    youtube_url: str,
    entry: dict,
    fps: float | None = None,
    media_resolution: str | None = None,
    video_duration_s: float | None = None,
) -> list[types.Part]:
    part_kwargs: dict = {"file_data": types.FileData(file_uri=youtube_url)}
    if fps:
        span = float(video_duration_s) if video_duration_s is not None else None
        eff = _effective_low_fps(span) if span is not None else fps
        part_kwargs["video_metadata"] = types.VideoMetadata(fps=eff)
    if media_resolution:
        part_kwargs["media_resolution"] = types.PartMediaResolution(level=media_resolution)
    return [types.Part(**part_kwargs), types.Part(text=_build_question_text(entry))]


def _choice_to_index(choice: str) -> int | None:
    if choice in LETTER_TO_INDEX:
        return LETTER_TO_INDEX[choice]
    if choice in DIGIT_TO_INDEX:
        return DIGIT_TO_INDEX[choice]
    return None


def _parse_answer_json(response_text: str) -> int | None:
    stripped = response_text.strip()
    if not stripped:
        return None
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    choice = data.get("choice")
    if isinstance(choice, int) and 1 <= choice <= 5:
        return choice - 1
    return None


def _parse_answer(response_text: str) -> int | None:
    stripped = response_text.strip()
    if stripped:
        first_line = stripped.splitlines()[0].strip()
        for token in (stripped, first_line):
            if len(token) == 1:
                idx = _choice_to_index(token.upper())
                if idx is not None:
                    return idx

    normalized = stripped.upper()
    for pattern in (
        r"FINAL ANSWER\s*:\s*\(?\s*([A-E1-5])\s*\)?",
        r"(?:ANSWER|CHOICE|OPTION)\s*[:\s]\s*\(?\s*([A-E1-5])\s*\)?",
    ):
        matches = re.findall(pattern, normalized)
        if matches:
            idx = _choice_to_index(matches[-1])
            if idx is not None:
                return idx

    standalone = re.findall(r"(?m)^\s*\(?\s*([A-E1-5])\s*\)?\s*$", normalized)
    if standalone:
        return _choice_to_index(standalone[-1])

    return None


def _predicted_id_from_response_text(response_text: str) -> int | None:
    parsed = _parse_answer_json(response_text)
    if parsed is not None:
        return parsed
    return _parse_answer(response_text)


async def answer_question(
    client: genai.Client,
    policy: Policy,
    youtube_url: str,
    video_id: str,
    segments: list[Segment],
    materialized: dict[int, SegmentMaterial],
    entry: dict,
    sem: asyncio.Semaphore,
    model: str = MODEL_NAME,
    pbar: Any = None,
    video_duration_s: float | None = None,
    routing_hash: str | None = None,
) -> tuple[int | None, str, str, TokenUsage]:
    """
    Answer a single question asynchronously.
    Returns (predicted_id, raw_response, raw_thoughts, usage).
    """
    # Keep prompt changes from silently reusing cached answers from older formats.
    # For LLM_ROUTED, include routing_hash so different directives don't share cache.
    policy_tag = f"{policy.value}_{ANSWER_PROMPT_CACHE_VERSION}"
    if routing_hash:
        policy_tag = f"{policy_tag}_{routing_hash}"
    ck = answer_cache_key(
        video_id,
        policy_tag,
        entry["key"],
    )
    cached = load_answer_cache(ck)
    if cached:
        usage = TokenUsage.from_dict(cached["usage"])
        usage.latency_s = 0.0
        if pbar: pbar.update(1)
        return (
            cached["predicted_id"],
            cached["raw_response"],
            cached.get("raw_thoughts", ""),
            usage,
        )

    if policy == Policy.RAW:
        parts = _build_video_parts(youtube_url, entry)
        contents = types.Content(parts=parts)
    elif policy == Policy.LOW_FPS:
        parts = _build_video_parts(
            youtube_url, entry, fps=LOW_FPS_RATE, video_duration_s=video_duration_s,
        )
        contents = types.Content(parts=parts)
    elif policy == Policy.LOW_RES:
        parts = _build_video_parts(youtube_url, entry, media_resolution=LOW_RES_MEDIA_RESOLUTION)
        contents = types.Content(parts=parts)
    elif policy in (Policy.VISUAL_DESCRIPTION, Policy.SUMMARY):
        prompt = _build_nonsegmented_text_prompt(materialized[0], entry)
        contents = prompt
    elif policy == Policy.TRANSCRIPT:
        prompt = _build_segmented_text_prompt(materialized, segments, entry)
        contents = prompt
    elif policy in (Policy.MIXED, Policy.LLM_ROUTED):
        parts = _build_mixed_parts(
            materialized, segments, youtube_url, entry, video_id, video_duration_s,
        )
        contents = types.Content(parts=parts)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    async with sem:
        t0 = time.monotonic()
        response = await with_retries_async(
            client.aio.models.generate_content,
            model=model,
            contents=contents,
            config=get_qa_config(model),
            label=f"qa({video_id},{entry['key']})",
        )
        latency_s = time.monotonic() - t0

    raw_text, raw_thoughts = split_main_and_thought_texts(response)
    usage = TokenUsage.from_response(response, latency_s=latency_s)
    predicted_id = _predicted_id_from_response_text(raw_text)

    save_answer_cache(ck, {
        "predicted_id": predicted_id,
        "raw_response": raw_text,
        "raw_thoughts": raw_thoughts,
        "usage": usage.to_dict(),
    })

    if pbar: pbar.update(1)
    return predicted_id, raw_text, raw_thoughts, usage
