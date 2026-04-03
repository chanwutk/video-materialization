import asyncio
import re
from typing import Any

from google import genai
from google.genai import types

from .cache import answer_cache_key, load_answer_cache, save_answer_cache
from .config import LOW_FPS_RATE, LOW_RES_MEDIA_RESOLUTION, MODEL_NAME
from .genai_config import GEMINI_GENERATE_CONTENT_CONFIG
from .genai_response import split_main_and_thought_texts
from .policies import Policy, SegmentMaterial
from .segmenter import Segment
from .tokens import TokenUsage


LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
DIGIT_TO_INDEX = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
ANSWER_PROMPT_CACHE_VERSION = "minerva-direct-v2"

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
        'Output the final answer in the format "Final Answer: (X)" where X is the correct digit choice.',
        "Do not output any explanation or the full answer text.",
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


def _build_mixed_parts(
    materialized: dict[int, SegmentMaterial],
    segments: list[Segment],
    youtube_url: str,
    entry: dict,
) -> list[types.Part]:
    parts = [types.Part(text=MATERIALIZATION_PREAMBLE)]
    for seg in segments:
        mat = materialized[seg.index]
        if mat.is_video:
            parts.append(types.Part(text=f"Segment {seg.index} ({seg.start_s}s-{seg.end_s}s) [{mat.material_type}]:"))
            part_kwargs: dict = {
                "file_data": types.FileData(file_uri=youtube_url),
                "video_metadata": types.VideoMetadata(
                    start_offset=f"{seg.start_s}s",
                    end_offset=f"{seg.end_s}s",
                ),
            }
            if mat.material_type == "low-res":
                part_kwargs["media_resolution"] = types.PartMediaResolution(level=LOW_RES_MEDIA_RESOLUTION)
            elif mat.material_type == "low-fps":
                part_kwargs["video_metadata"] = types.VideoMetadata(
                    start_offset=f"{seg.start_s}s",
                    end_offset=f"{seg.end_s}s",
                    fps=LOW_FPS_RATE,
                )
            parts.append(types.Part(**part_kwargs))
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
) -> list[types.Part]:
    part_kwargs: dict = {"file_data": types.FileData(file_uri=youtube_url)}
    if fps:
        part_kwargs["video_metadata"] = types.VideoMetadata(fps=fps)
    if media_resolution:
        part_kwargs["media_resolution"] = types.PartMediaResolution(level=media_resolution)
    return [types.Part(**part_kwargs), types.Part(text=_build_question_text(entry))]


def _parse_answer(response_text: str) -> int | None:
    normalized = response_text.strip().upper()
    for pattern in (
        r"FINAL ANSWER\s*:\s*\(?\s*([A-E1-5])\s*\)?",
        r"(?:ANSWER|CHOICE|OPTION)\s*[:\s]\s*\(?\s*([A-E1-5])\s*\)?",
    ):
        matches = re.findall(pattern, normalized)
        if matches:
            choice = matches[-1]
            if choice in LETTER_TO_INDEX:
                return LETTER_TO_INDEX[choice]
            return DIGIT_TO_INDEX[choice]

    standalone = re.findall(r"(?m)^\s*\(?\s*([A-E1-5])\s*\)?\s*$", normalized)
    if standalone:
        choice = standalone[-1]
        if choice in LETTER_TO_INDEX:
            return LETTER_TO_INDEX[choice]
        return DIGIT_TO_INDEX[choice]

    return None


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
) -> tuple[int | None, str, str, TokenUsage]:
    """
    Answer a single question asynchronously.
    Returns (predicted_id, raw_response, raw_thoughts, usage).
    """
    # Keep prompt changes from silently reusing cached answers from older formats.
    ck = answer_cache_key(
        video_id,
        f"{policy.value}_{ANSWER_PROMPT_CACHE_VERSION}",
        entry["key"],
    )
    cached = load_answer_cache(ck)
    if cached:
        if pbar: pbar.update(1)
        return (
            cached["predicted_id"],
            cached["raw_response"],
            cached.get("raw_thoughts", ""),
            TokenUsage.from_dict(cached["usage"]),
        )

    if policy == Policy.RAW:
        parts = _build_video_parts(youtube_url, entry)
        contents = types.Content(parts=parts)
    elif policy == Policy.LOW_FPS:
        parts = _build_video_parts(youtube_url, entry, fps=LOW_FPS_RATE)
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
    elif policy == Policy.MIXED:
        parts = _build_mixed_parts(materialized, segments, youtube_url, entry)
        contents = types.Content(parts=parts)
    else:
        raise ValueError(f"Unknown policy: {policy}")

    async with sem:
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=GEMINI_GENERATE_CONTENT_CONFIG,
        )

    raw_text, raw_thoughts = split_main_and_thought_texts(response)
    usage = TokenUsage.from_response(response)
    predicted_id = _parse_answer(raw_text)

    save_answer_cache(ck, {
        "predicted_id": predicted_id,
        "raw_response": raw_text,
        "raw_thoughts": raw_thoughts,
        "usage": usage.to_dict(),
    })

    if pbar: pbar.update(1)
    return predicted_id, raw_text, raw_thoughts, usage
