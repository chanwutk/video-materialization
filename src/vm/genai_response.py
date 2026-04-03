"""Split Gemini `GenerateContentResponse` into main answer text vs thought parts."""

from __future__ import annotations


def split_main_and_thought_texts(response) -> tuple[str, str]:
    """
    Uses candidate parts with `thought=True` vs `thought=False`.
    Falls back to `response.text` for main when no non-thought parts have text.
    """
    main_chunks: list[str] = []
    thought_chunks: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            for part in parts:
                raw = getattr(part, "text", None) or ""
                s = str(raw).strip()
                if not s:
                    continue
                if getattr(part, "thought", None):
                    thought_chunks.append(s)
                else:
                    main_chunks.append(s)
    main = "\n\n".join(main_chunks) if main_chunks else ""
    thoughts = "\n\n".join(thought_chunks)
    if not main:
        main = (getattr(response, "text", None) or "").strip()
    return main, thoughts
