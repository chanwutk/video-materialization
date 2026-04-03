"""Shared Gemini `GenerateContentConfig`: no tools, safety off, greedy / deterministic sampling."""

from google.genai import types

# Fixed seed improves repeatability when the API honors it (not guaranteed across model updates).
GEMINI_SAMPLING_SEED = 42

_HARM_CATEGORIES = tuple(
    c
    for c in types.HarmCategory
    if c is not types.HarmCategory.HARM_CATEGORY_UNSPECIFIED
)

GEMINI_GENERATE_CONTENT_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    candidate_count=1,
    seed=GEMINI_SAMPLING_SEED,
    thinking_config=types.ThinkingConfig(
        thinking_budget=-1,
        thinking_level=types.ThinkingLevel.HIGH,
        include_thoughts=True,
    ),
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
