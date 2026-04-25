"""LLM-based segment routing: decide materialization method per segment."""

import hashlib
import json

from google import genai
from google.genai import types

from .cache import load_builder_cache, cache_key
from .config import ROUTER_MODEL_NAME, ROUTING_CACHE_DIR
from .genai_config import GEMINI_SAMPLING_SEED, _HARM_CATEGORIES
from .retry import with_retries
from .segmenter import Segment

VALID_DECISIONS = {"TRANSCRIPT", "SUMMARY", "LOW_FPS", "SKIP"}

ROUTING_TEMPLATE = """\
You are a video segment routing system. For each segment of a video, decide the best materialization method to represent it for downstream question-answering tasks.

Available methods:
- TRANSCRIPT: Use the verbatim speech transcription. Best when the segment contains important dialogue or narration.
- SUMMARY: Use a text summary of the segment. Best when the segment has moderate content that can be condensed into text.
- LOW_FPS: Keep the original video at low frame rate (0.2 fps). Best when visual details are critical and text alone cannot capture them.
- SKIP: Skip this segment entirely. Use when the segment contains no relevant content (e.g., silence, static screen, credits).

{directive}

The video has {n_segments} segments. Below is metadata for each segment.

{segments_info}

Respond with a JSON array of exactly {n_segments} decisions, one per segment in order.
Each decision must be one of: "TRANSCRIPT", "SUMMARY", "LOW_FPS", "SKIP".
"""

ROUTING_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "string",
        "enum": ["TRANSCRIPT", "SUMMARY", "LOW_FPS", "SKIP"],
    },
}

ROUTER_GENERATE_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    candidate_count=1,
    seed=GEMINI_SAMPLING_SEED,
    response_mime_type="application/json",
    response_json_schema=ROUTING_RESPONSE_SCHEMA,
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=types.FunctionCallingConfigMode.NONE,
        ),
    ),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    safety_settings=[
        types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.OFF)
        for c in _HARM_CATEGORIES
    ],
)

# Default seed directive (based on the existing MIXED heuristic).
SEED_DIRECTIVE = """\
Decision guidelines:
- If the transcript has more than 30 words of substantive speech (not just filler), use TRANSCRIPT.
- If the segment has notable visual activity (the summary describes actions, movements, or changes) or any speech at all, use LOW_FPS to preserve visual context.
- If the segment is mostly quiet with minimal visual change, use SUMMARY to save tokens.
- If the segment appears to be dead air, a static title card, or end credits with no informational content, use SKIP."""


def _directive_hash(directive: str) -> str:
    return hashlib.sha256(directive.encode()).hexdigest()[:16]


def _build_segments_info(
    video_id: str,
    segments: list[Segment],
) -> str:
    """Build a text block describing each segment using cached materializations."""
    lines = []
    for seg in segments:
        transcript_cache = load_builder_cache(cache_key(video_id, seg.index, "transcript"))
        summary_cache = load_builder_cache(cache_key(video_id, seg.index, "summary"))
        transcript = transcript_cache["text"] if transcript_cache else "[NOT AVAILABLE]"
        summary = summary_cache["text"] if summary_cache else "[NOT AVAILABLE]"
        t_words = len(transcript.split()) if transcript != "[NOT AVAILABLE]" else 0
        s_words = len(summary.split()) if summary != "[NOT AVAILABLE]" else 0
        lines.append(
            f"Segment {seg.index} ({seg.start_s}s-{seg.end_s}s):\n"
            f"  Transcript ({t_words} words): {transcript[:300]}\n"
            f"  Summary ({s_words} words): {summary[:300]}"
        )
    return "\n\n".join(lines)


def _load_routing_cache(video_id: str, d_hash: str) -> list[str] | None:
    ROUTING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = video_id.replace("/", "_").replace("\\", "_")
    path = ROUTING_CACHE_DIR / f"{safe_id}_{d_hash}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_routing_cache(video_id: str, d_hash: str, decisions: list[str]) -> None:
    ROUTING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = video_id.replace("/", "_").replace("\\", "_")
    path = ROUTING_CACHE_DIR / f"{safe_id}_{d_hash}.json"
    path.write_text(json.dumps(decisions))


def route_video_segments(
    client: genai.Client,
    video_id: str,
    segments: list[Segment],
    directive: str,
    model: str = ROUTER_MODEL_NAME,
) -> list[str]:
    """
    Route all segments of a video using the LLM router.
    Returns a list of decisions (one per segment), each in VALID_DECISIONS.
    Uses disk cache keyed by (video_id, directive_hash).
    """
    d_hash = _directive_hash(directive)

    cached = _load_routing_cache(video_id, d_hash)
    if cached is not None and len(cached) == len(segments):
        return cached

    segments_info = _build_segments_info(video_id, segments)
    prompt = ROUTING_TEMPLATE.format(
        directive=directive,
        n_segments=len(segments),
        segments_info=segments_info,
    )

    response = with_retries(
        client.models.generate_content,
        model=model,
        contents=prompt,
        config=ROUTER_GENERATE_CONFIG,
        label=f"router({video_id})",
    )

    raw = response.text.strip()
    decisions = json.loads(raw)

    # Validate and fix
    if len(decisions) != len(segments):
        # Pad or truncate to match
        if len(decisions) < len(segments):
            decisions.extend(["SUMMARY"] * (len(segments) - len(decisions)))
        else:
            decisions = decisions[:len(segments)]

    for i, d in enumerate(decisions):
        if d not in VALID_DECISIONS:
            decisions[i] = "SUMMARY"

    _save_routing_cache(video_id, d_hash, decisions)
    return decisions
