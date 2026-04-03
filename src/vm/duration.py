import json
import subprocess

from .config import DURATIONS_CACHE, CACHE_DIR


def _load_duration_cache() -> dict[str, float]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if DURATIONS_CACHE.exists():
        return json.loads(DURATIONS_CACHE.read_text())
    return {}


def _save_duration_cache(cache: dict[str, float]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DURATIONS_CACHE.write_text(json.dumps(cache, indent=2))


def get_video_duration(video_id: str) -> float | None:
    cache = _load_duration_cache()
    if video_id in cache:
        return cache[video_id]

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
    duration = float(info["duration"])
    cache[video_id] = duration
    _save_duration_cache(cache)
    return duration


def get_durations_for_videos(video_ids: list[str]) -> dict[str, float]:
    durations = {}
    for vid in video_ids:
        d = get_video_duration(vid)
        if d is not None:
            durations[vid] = d
        else:
            print(f"  Skipping {vid} (unavailable)")
    return durations
