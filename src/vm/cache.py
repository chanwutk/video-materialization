import json
from pathlib import Path

from .config import BUILDERS_CACHE_DIR, ANSWERS_CACHE_DIR


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cache_key(video_id: str, segment_index: int, builder_type: str) -> str:
    safe_id = video_id.replace("/", "_").replace("\\", "_")
    return f"{safe_id}_seg{segment_index:02d}_{builder_type}"


def load_builder_cache(key: str) -> dict | None:
    _ensure_dir(BUILDERS_CACHE_DIR)
    path = BUILDERS_CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_builder_cache(key: str, data: dict) -> None:
    _ensure_dir(BUILDERS_CACHE_DIR)
    path = BUILDERS_CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(data, indent=2))


def answer_cache_key(video_id: str, policy: str, question_key: str) -> str:
    safe_id = video_id.replace("/", "_").replace("\\", "_")
    safe_qk = question_key.replace("/", "_").replace("\\", "_")
    return f"{safe_id}_{policy}_{safe_qk}"


def load_answer_cache(key: str) -> dict | None:
    _ensure_dir(ANSWERS_CACHE_DIR)
    path = ANSWERS_CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_answer_cache(key: str, data: dict) -> None:
    _ensure_dir(ANSWERS_CACHE_DIR)
    path = ANSWERS_CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(data, indent=2))
