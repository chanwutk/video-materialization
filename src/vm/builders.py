import time

from google import genai
from google.genai import types

from .cache import cache_key, load_builder_cache, save_builder_cache
from .config import API_CALL_DELAY_S, MODEL_NAME
from .segmenter import Segment
from .tokens import TokenUsage


def _make_video_part(youtube_url: str, segment: Segment) -> types.Part:
    return types.Part(
        file_data=types.FileData(file_uri=youtube_url),
        video_metadata=types.VideoMetadata(
            start_offset=f"{segment.start_s}s",
            end_offset=f"{segment.end_s}s",
        ),
    )


def _call_gemini(
    client: genai.Client,
    youtube_url: str,
    segment: Segment,
    prompt: str,
    model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    video_part = _make_video_part(youtube_url, segment)
    text_part = types.Part(text=prompt)

    response = client.models.generate_content(
        model=model,
        contents=types.Content(parts=[video_part, text_part]),
    )
    time.sleep(API_CALL_DELAY_S)
    return response.text, TokenUsage.from_response(response)


def build_transcript(
    client: genai.Client, video_id: str, youtube_url: str, segment: Segment, model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    key = cache_key(video_id, segment.index, "transcript")
    cached = load_builder_cache(key)
    if cached:
        return cached["text"], TokenUsage.from_dict(cached["usage"])

    text, usage = _call_gemini(
        client, youtube_url, segment,
        "Transcribe all speech in this video segment verbatim. If there is no speech, respond with [NO SPEECH].",
        model=model,
    )
    save_builder_cache(key, {"text": text, "usage": usage.to_dict()})
    return text, usage


def build_keyframes(
    client: genai.Client, video_id: str, youtube_url: str, segment: Segment, model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    key = cache_key(video_id, segment.index, "keyframes")
    cached = load_builder_cache(key)
    if cached:
        return cached["text"], TokenUsage.from_dict(cached["usage"])

    text, usage = _call_gemini(
        client, youtube_url, segment,
        "Describe the key visual scenes in this video segment. For each distinct scene, provide a one-sentence description of what is shown.",
        model=model,
    )
    save_builder_cache(key, {"text": text, "usage": usage.to_dict()})
    return text, usage


def build_summary(
    client: genai.Client, video_id: str, youtube_url: str, segment: Segment, model: str = MODEL_NAME,
) -> tuple[str, TokenUsage]:
    key = cache_key(video_id, segment.index, "summary")
    cached = load_builder_cache(key)
    if cached:
        return cached["text"], TokenUsage.from_dict(cached["usage"])

    text, usage = _call_gemini(
        client, youtube_url, segment,
        "Provide a concise summary of this video segment in 2-3 sentences, covering both visual content and any speech.",
        model=model,
    )
    save_builder_cache(key, {"text": text, "usage": usage.to_dict()})
    return text, usage


BUILDER_FNS = {
    "transcript": build_transcript,
    "keyframes": build_keyframes,
    "summary": build_summary,
}
