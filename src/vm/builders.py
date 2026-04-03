import time

from google import genai
from google.genai import types

from .cache import (
    cache_key, whole_video_cache_key,
    load_builder_cache, save_builder_cache,
)
from .config import API_CALL_DELAY_S, MODEL_NAME
from .segmenter import Segment
from .tokens import TokenUsage


def _make_segment_video_part(youtube_url: str, segment: Segment) -> types.Part:
    return types.Part(
        file_data=types.FileData(file_uri=youtube_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{segment.start_s}s",
            end_offset=f"{segment.end_s}s",
        ),
    )


def _make_whole_video_part(youtube_url: str) -> types.Part:
    return types.Part(
        file_data=types.FileData(file_uri=youtube_url),
    )


def _call_gemini(
    client: genai.Client,
    video_part: types.Part,
    prompt: str,
    model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    text_part = types.Part(text=prompt)
    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=[video_part, text_part]),
    )
    time.sleep(API_CALL_DELAY_S)
    return response.text, TokenUsage.from_response(response)


# --- Per-segment builders (used by transcript policy and mixed policy) ---

def build_transcript(
    client: genai.Client, video_id: str, youtube_url: str, segment: Segment, model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    key = cache_key(video_id, segment.index, "transcript")
    cached = load_builder_cache(key)
    if cached:
        return cached["text"], TokenUsage.from_dict(cached["usage"])

    video_part = _make_segment_video_part(youtube_url, segment)
    text, usage = _call_gemini(
        client, video_part,
        "Transcribe all speech in this video segment verbatim. If there is no speech, respond with [NO SPEECH].",
        model=model,
    )
    save_builder_cache(key, {"text": text, "usage": usage.to_dict()})
    return text, usage


def build_segment_summary(
    client: genai.Client, video_id: str, youtube_url: str, segment: Segment, model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    """Per-segment summary, used by the mixed policy."""
    key = cache_key(video_id, segment.index, "summary")
    cached = load_builder_cache(key)
    if cached:
        return cached["text"], TokenUsage.from_dict(cached["usage"])

    video_part = _make_segment_video_part(youtube_url, segment)
    text, usage = _call_gemini(
        client, video_part,
        "Provide a concise summary of this video segment in 2-3 sentences, covering both visual content and any speech.",
        model=model,
    )
    save_builder_cache(key, {"text": text, "usage": usage.to_dict()})
    return text, usage


# --- Whole-video builders (used by standalone policies) ---

def build_visual_description(
    client: genai.Client, video_id: str, youtube_url: str, model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    key = whole_video_cache_key(video_id, "visual-description")
    cached = load_builder_cache(key)
    if cached:
        return cached["text"], TokenUsage.from_dict(cached["usage"])

    video_part = _make_whole_video_part(youtube_url)
    text, usage = _call_gemini(
        client, video_part,
        "Describe the key visual scenes in this video in chronological order. For each distinct scene, provide a one-sentence description of what is shown.",
        model=model,
    )
    save_builder_cache(key, {"text": text, "usage": usage.to_dict()})
    return text, usage


def build_whole_summary(
    client: genai.Client, video_id: str, youtube_url: str, model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    key = whole_video_cache_key(video_id, "summary")
    cached = load_builder_cache(key)
    if cached:
        return cached["text"], TokenUsage.from_dict(cached["usage"])

    video_part = _make_whole_video_part(youtube_url)
    text, usage = _call_gemini(
        client, video_part,
        "Provide a concise summary of this video covering both visual content and any speech.",
        model=model,
    )
    save_builder_cache(key, {"text": text, "usage": usage.to_dict()})
    return text, usage
