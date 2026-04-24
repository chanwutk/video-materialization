import asyncio
import time
from typing import Any

from google import genai
from google.genai import types

from .cache import (
    cache_key, whole_video_cache_key,
    load_builder_cache, save_builder_cache,
)
from .config import MODEL_NAME
from .genai_config import get_builder_config
from .genai_response import split_main_and_thought_texts
from .segmenter import Segment
from .tokens import TokenUsage


def _usage_from_cache(cached: dict) -> TokenUsage:
    usage = TokenUsage.from_dict(cached["usage"])
    usage.latency_s = 0.0
    return usage


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


async def _call_gemini(
    client: genai.Client,
    video_part: types.Part,
    prompt: str,
    sem: asyncio.Semaphore,
    model: str = MODEL_NAME,
) -> tuple[str, TokenUsage, str]:
    text_part = types.Part(text=prompt)
    async with sem:
        t0 = time.monotonic()
        response = await client.aio.models.generate_content(
            model=model,
            contents=types.Content(parts=[video_part, text_part]),
            config=get_builder_config(model),
        )
        latency_s = time.monotonic() - t0
    main_text, thoughts_text = split_main_and_thought_texts(response)
    return main_text, TokenUsage.from_response(response, latency_s=latency_s), thoughts_text


# --- Per-segment builders (used by transcript policy and mixed policy) ---

async def build_transcript(
    client: genai.Client, video_id: str, youtube_url: str, segment: Segment,
    sem: asyncio.Semaphore, model: str = MODEL_NAME, pbar: Any = None,
) -> tuple[str, TokenUsage]:
    key = cache_key(video_id, segment.index, "transcript")
    cached = load_builder_cache(key)
    if cached:
        if pbar: pbar.update(1)
        return cached["text"], _usage_from_cache(cached)

    video_part = _make_segment_video_part(youtube_url, segment)
    text, usage, thoughts = await _call_gemini(
        client, video_part,
        "Transcribe all speech in this video segment verbatim. If there is no speech, respond with [NO SPEECH].",
        sem, model=model,
    )
    save_builder_cache(key, {"text": text, "thoughts": thoughts, "usage": usage.to_dict()})
    if pbar: pbar.update(1)
    return text, usage


async def build_segment_summary(
    client: genai.Client, video_id: str, youtube_url: str, segment: Segment,
    sem: asyncio.Semaphore, model: str = MODEL_NAME, pbar: Any = None,
) -> tuple[str, TokenUsage]:
    """Per-segment summary, used by the mixed policy."""
    key = cache_key(video_id, segment.index, "summary")
    cached = load_builder_cache(key)
    if cached:
        if pbar: pbar.update(1)
        return cached["text"], _usage_from_cache(cached)

    video_part = _make_segment_video_part(youtube_url, segment)
    text, usage, thoughts = await _call_gemini(
        client, video_part,
        "Provide a concise summary of this video segment in 2-3 sentences, covering both visual content and any speech.",
        sem, model=model,
    )
    save_builder_cache(key, {"text": text, "thoughts": thoughts, "usage": usage.to_dict()})
    if pbar: pbar.update(1)
    return text, usage


# --- Whole-video builders (used by standalone policies) ---

async def build_visual_description(
    client: genai.Client, video_id: str, youtube_url: str,
    sem: asyncio.Semaphore, model: str = MODEL_NAME, pbar: Any = None,
) -> tuple[str, TokenUsage]:
    key = whole_video_cache_key(video_id, "visual-description")
    cached = load_builder_cache(key)
    if cached:
        if pbar: pbar.update(1)
        return cached["text"], _usage_from_cache(cached)

    video_part = _make_whole_video_part(youtube_url)
    text, usage, thoughts = await _call_gemini(
        client, video_part,
        "Describe the key visual scenes in this video in chronological order. For each distinct scene, provide a one-sentence description of what is shown.",
        sem, model=model,
    )
    save_builder_cache(key, {"text": text, "thoughts": thoughts, "usage": usage.to_dict()})
    if pbar: pbar.update(1)
    return text, usage


async def build_whole_summary(
    client: genai.Client, video_id: str, youtube_url: str,
    sem: asyncio.Semaphore, model: str = MODEL_NAME, pbar: Any = None,
) -> tuple[str, TokenUsage]:
    key = whole_video_cache_key(video_id, "summary")
    cached = load_builder_cache(key)
    if cached:
        if pbar: pbar.update(1)
        return cached["text"], _usage_from_cache(cached)

    video_part = _make_whole_video_part(youtube_url)
    text, usage, thoughts = await _call_gemini(
        client, video_part,
        "Provide a concise summary of this video covering both visual content and any speech.",
        sem, model=model,
    )
    save_builder_cache(key, {"text": text, "thoughts": thoughts, "usage": usage.to_dict()})
    if pbar: pbar.update(1)
    return text, usage
