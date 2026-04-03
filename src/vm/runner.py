import re
import time

from google import genai
from google.genai import types

from .cache import answer_cache_key, load_answer_cache, save_answer_cache
from .config import API_CALL_DELAY_S, MODEL_NAME
from .policies import Policy
from .segmenter import Segment
from .tokens import TokenUsage


LETTER_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _build_question_text(entry: dict) -> str:
    q = entry["question"]
    choices = [
        f"A) {entry['answer_choice_0']}",
        f"B) {entry['answer_choice_1']}",
        f"C) {entry['answer_choice_2']}",
        f"D) {entry['answer_choice_3']}",
        f"E) {entry['answer_choice_4']}",
    ]
    return f"Question: {q}\n" + "\n".join(choices) + "\n\nRespond with only the letter of the correct answer."


def _build_materialized_prompt(
    materialized: dict[int, str],
    segments: list[Segment],
    entry: dict,
) -> str:
    lines = ["Below is pre-computed context for a video, organized by segment.\n"]
    for seg in segments:
        text = materialized.get(seg.index, "[NO DATA]")
        lines.append(f"Segment {seg.index} ({seg.start_s}s-{seg.end_s}s): {text}\n")
    lines.append("")
    lines.append(_build_question_text(entry))
    return "\n".join(lines)


def _build_raw_parts(
    youtube_url: str,
    segments: list[Segment],
    entry: dict,
) -> list[types.Part]:
    parts = []
    for seg in segments:
        parts.append(types.Part(
            file_data=types.FileData(file_uri=youtube_url),
            video_metadata=types.VideoMetadata(
                start_offset=f"{seg.start_s}s",
                end_offset=f"{seg.end_s}s",
            ),
        ))
    parts.append(types.Part(text=_build_question_text(entry)))
    return parts


def _parse_answer(response_text: str) -> int | None:
    match = re.search(r"[A-E]", response_text.strip().upper())
    if match:
        return LETTER_TO_INDEX[match.group()]
    return None


def answer_question(
    client: genai.Client,
    policy: Policy,
    youtube_url: str,
    video_id: str,
    segments: list[Segment],
    materialized: dict[int, str | None],
    entry: dict,
    model: str = MODEL_NAME,
) -> tuple[int | None, str, TokenUsage]:
    """
    Answer a single question.

    Returns: (predicted_id, raw_response_text, token_usage)
    """
    # Check cache
    ck = answer_cache_key(video_id, policy.value, entry["key"])
    cached = load_answer_cache(ck)
    if cached:
        return cached["predicted_id"], cached["raw_response"], TokenUsage.from_dict(cached["usage"])

    if policy == Policy.RAW:
        parts = _build_raw_parts(youtube_url, segments, entry)
        response = client.models.generate_content(
            model=model,
            contents=types.Content(parts=parts),
        )
    else:
        prompt = _build_materialized_prompt(materialized, segments, entry)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )

    time.sleep(API_CALL_DELAY_S)

    raw_text = response.text
    usage = TokenUsage.from_response(response)
    predicted_id = _parse_answer(raw_text)

    # Cache result
    save_answer_cache(ck, {
        "predicted_id": predicted_id,
        "raw_response": raw_text,
        "usage": usage.to_dict(),
    })

    return predicted_id, raw_text, usage
