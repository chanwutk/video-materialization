"""Shared Gemini `GenerateContentConfig`: no tools, safety off, greedy / deterministic sampling."""

from typing import Any

from google.genai import types

# Fixed seed improves repeatability when the API honors it (not guaranteed across model updates).
GEMINI_SAMPLING_SEED = 42

# generativelanguage v1beta rejects IMAGE_* and JAILBREAK here; SDK lists more than the API accepts.
_HARM_CATEGORIES = (
    types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
)

# Models that support ThinkingConfig with thinking_level.
_THINKING_SUPPORTED_MODELS = {"gemini-3.1-pro-preview"}

_BASE_CONFIG_KWARGS: dict[str, Any] = dict(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    candidate_count=1,
    seed=GEMINI_SAMPLING_SEED,
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

# Default config with thinking (for gemini-3.1-pro-preview).
GEMINI_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    **_BASE_CONFIG_KWARGS,
    thinking_config=types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.LOW,
        include_thoughts=True,
    ),
)

# Config without thinking (for models that don't support it).
GEMINI_GENERATE_CONTENT_CONFIG_NO_THINKING = types.GenerateContentConfig(
    **_BASE_CONFIG_KWARGS,
)

# Video QA: JSON object {"choice": 1..5} per Gemini structured output (JSON Schema).
# https://ai.google.dev/gemini-api/docs/structured-output
QA_CHOICE_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "choice": {
            "type": "integer",
            "enum": [1, 2, 3, 4, 5],
            "description": "The correct multiple-choice option number (1-5).",
        },
    },
    "required": ["choice"],
}

GEMINI_QA_GENERATE_CONTENT_CONFIG = GEMINI_GENERATE_CONTENT_CONFIG.model_copy(
    update={
        "response_mime_type": "application/json",
        "response_json_schema": QA_CHOICE_RESPONSE_JSON_SCHEMA,
    },
)

GEMINI_QA_GENERATE_CONTENT_CONFIG_NO_THINKING = GEMINI_GENERATE_CONTENT_CONFIG_NO_THINKING.model_copy(
    update={
        "response_mime_type": "application/json",
        "response_json_schema": QA_CHOICE_RESPONSE_JSON_SCHEMA,
    },
)


def get_qa_config(model: str) -> types.GenerateContentConfig:
    """Return the appropriate QA config based on model thinking support."""
    if model in _THINKING_SUPPORTED_MODELS:
        return GEMINI_QA_GENERATE_CONTENT_CONFIG
    return GEMINI_QA_GENERATE_CONTENT_CONFIG_NO_THINKING


def get_builder_config(model: str) -> types.GenerateContentConfig:
    """Return the appropriate builder config based on model thinking support."""
    if model in _THINKING_SUPPORTED_MODELS:
        return GEMINI_GENERATE_CONTENT_CONFIG
    return GEMINI_GENERATE_CONTENT_CONFIG_NO_THINKING
