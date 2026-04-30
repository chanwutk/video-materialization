import asyncio
import sys
import time
from typing import Any

from google import genai
from google.genai import types

from .cache import (
    cache_key, whole_video_cache_key,
    load_builder_cache, save_builder_cache,
)
from .config import (
    LLOVI_CAPTION_PROMPT,
    LLOVI_CAPTIONER_MODEL,
    LLOVI_CLIP_FPS,
    LLOVI_CLIP_LENGTH_S,
    LLOVI_NO_CAPTION_PLACEHOLDER,
    MODEL_NAME,
    VIDEO_DURATION_CLIP_SLACK_S,
)
from .genai_config import get_builder_config
from .retry import with_retries_async
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
        response = await with_retries_async(
            client.aio.models.generate_content,
            model=model,
            contents=types.Content(parts=[video_part, text_part]),
            config=get_builder_config(model),
            label="builder",
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


# --- LLoVi (Zhang et al. 2023) baseline: dense per-clip captions concatenated as text. ---

def _llovi_clip_cache_key(video_id: str, clip_start_s: int) -> str:
    """One cache slot per (video, clip start). Dense and sparse share clips at t=0,8,16,..."""
    safe_id = video_id.replace("/", "_").replace("\\", "_")
    return f"{safe_id}_llovi_clip_t{clip_start_s:05d}s"


def llovi_clip_starts(video_duration_s: float, stride_s: int) -> list[int]:
    """
    Clip start offsets for LLoVi. Each clip is LLOVI_CLIP_LENGTH_S long, stepping by stride.
    Drops clips whose end would exceed (duration - VIDEO_DURATION_CLIP_SLACK_S) to avoid
    INVALID_ARGUMENT from Gemini on the tail.
    """
    safe_upper = int(video_duration_s) - VIDEO_DURATION_CLIP_SLACK_S
    if safe_upper < LLOVI_CLIP_LENGTH_S:
        return []
    last_start = safe_upper - LLOVI_CLIP_LENGTH_S
    return list(range(0, last_start + 1, stride_s))


def _make_llovi_clip_video_part(youtube_url: str, clip_start_s: int) -> types.Part:
    return types.Part(
        file_data=types.FileData(file_uri=youtube_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{clip_start_s}s",
            end_offset=f"{clip_start_s + LLOVI_CLIP_LENGTH_S}s",
            fps=LLOVI_CLIP_FPS,
        ),
    )


async def build_llovi_clip_caption(
    client: genai.Client, video_id: str, youtube_url: str, clip_start_s: int,
    sem: asyncio.Semaphore, model: str = LLOVI_CAPTIONER_MODEL, pbar: Any = None,
) -> tuple[str, TokenUsage]:
    """
    Caption a single 1s clip starting at clip_start_s. Cached per (video, clip_start),
    so dense and sparse streams share captions at coincident timestamps.
    """
    key = _llovi_clip_cache_key(video_id, clip_start_s)
    cached = load_builder_cache(key)
    if cached:
        if pbar: pbar.update(1)
        return cached["text"], _usage_from_cache(cached)

    video_part = _make_llovi_clip_video_part(youtube_url, clip_start_s)
    try:
        text, usage, thoughts = await _call_gemini(
            client, video_part, LLOVI_CAPTION_PROMPT, sem, model=model,
        )
    except Exception as exc:
        # Skip-and-mark: persistent failure on one clip should not kill the stream.
        print(
            f"[llovi] caption failed for {video_id} clip t={clip_start_s}s: {exc!r}",
            file=sys.stderr,
        )
        usage = TokenUsage()
        save_builder_cache(key, {
            "text": LLOVI_NO_CAPTION_PLACEHOLDER,
            "thoughts": "",
            "usage": usage.to_dict(),
            "error": repr(exc),
        })
        if pbar: pbar.update(1)
        return LLOVI_NO_CAPTION_PLACEHOLDER, usage

    save_builder_cache(key, {"text": text, "thoughts": thoughts, "usage": usage.to_dict()})
    if pbar: pbar.update(1)
    return text, usage


async def build_llovi_stream(
    client: genai.Client, video_id: str, youtube_url: str,
    video_duration_s: float, stride_s: int,
    sem: asyncio.Semaphore, model: str = LLOVI_CAPTIONER_MODEL, pbar: Any = None,
) -> tuple[str, list[TokenUsage]]:
    """
    Build a LLoVi caption stream for the whole video at the given stride.
    Returns concatenated `[Xs-Ys] caption` lines plus the per-clip token usages.
    """
    starts = llovi_clip_starts(video_duration_s, stride_s)
    if not starts:
        return "", []

    tasks = [
        build_llovi_clip_caption(client, video_id, youtube_url, s, sem, model=model, pbar=pbar)
        for s in starts
    ]
    results = await asyncio.gather(*tasks)

    lines = []
    usages: list[TokenUsage] = []
    for start, (text, usage) in zip(starts, results):
        end = start + LLOVI_CLIP_LENGTH_S
        clean = text.strip().replace("\n", " ")
        lines.append(f"[{start}s-{end}s] {clean}")
        usages.append(usage)
    return "\n".join(lines), usages
