import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

from .config import DURATIONS_CACHE, CACHE_DIR

# Network-bound yt-dlp calls; cap to avoid huge process fan-out.
_DURATION_FETCH_WORKERS = 32


def _load_duration_cache() -> dict[str, float]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if DURATIONS_CACHE.exists():
        return json.loads(DURATIONS_CACHE.read_text())
    return {}


def read_duration_cache() -> dict[str, float]:
    """Cached durations only; no network. Used for tie-breaking video selection."""
    return _load_duration_cache()


def _save_duration_cache(cache: dict[str, float]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DURATIONS_CACHE.write_text(json.dumps(cache, indent=2))


def _fetch_duration_yt_dlp(video_id: str) -> float | None:
    """Query duration via yt-dlp; no cache I/O."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        result = subprocess.run(
            ["yt-dlp", "--dump-json", "--no-download", url],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        print(f"  Timeout fetching duration for {video_id}")
        return None

    if result.returncode != 0:
        print(f"  Failed to fetch duration for {video_id}: {result.stderr[:200]}")
        return None

    info = json.loads(result.stdout)
    return float(info["duration"])


def get_video_duration(video_id: str) -> float | None:
    cache = _load_duration_cache()
    if video_id in cache:
        return cache[video_id]

    duration = _fetch_duration_yt_dlp(video_id)
    if duration is None:
        return None
    cache[video_id] = duration
    _save_duration_cache(cache)
    return duration


def get_durations_for_videos(video_ids: list[str]) -> dict[str, float]:
    cache = _load_duration_cache()
    durations: dict[str, float] = {}
    to_fetch: list[str] = []

    for vid in video_ids:
        if vid in cache:
            durations[vid] = cache[vid]
        else:
            to_fetch.append(vid)

    if to_fetch:
        workers = min(_DURATION_FETCH_WORKERS, len(to_fetch))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fetched = list(ex.map(_fetch_duration_yt_dlp, to_fetch))
        for vid, d in zip(to_fetch, fetched):
            if d is not None:
                durations[vid] = d
                cache[vid] = d
            else:
                print(f"  Skipping {vid} (unavailable)")
        _save_duration_cache(cache)

    return durations
