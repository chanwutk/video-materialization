import json
import random
from collections import Counter

import httpx

from .config import MINERVA_URL, MINERVA_LOCAL, DATA_DIR, GEPA_TRAIN_TEST_SEED, GEPA_TRAIN_SIZE


def download_minerva() -> list[dict]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MINERVA_LOCAL.exists():
        return json.loads(MINERVA_LOCAL.read_text())
    print(f"Downloading MINERVA dataset from {MINERVA_URL}...")
    resp = httpx.get(MINERVA_URL, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    MINERVA_LOCAL.write_bytes(resp.content)
    return resp.json()


def group_by_video(entries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        vid = entry["video_id"]
        grouped.setdefault(vid, []).append(entry)
    return grouped


def select_top_k(
    grouped: dict[str, list[dict]],
    k: int,
    *,
    duration_hint: dict[str, float] | None = None,
) -> dict[str, list[dict]]:
    def sort_key(item: tuple[str, list[dict]]) -> tuple[int, float]:
        vid, qs = item
        n = len(qs)
        dur = duration_hint.get(vid, 0.0) if duration_hint else 0.0
        return (n, dur)

    sorted_videos = sorted(grouped.items(), key=sort_key, reverse=True)
    selected = dict(sorted_videos[:k])

    counts = Counter(len(qs) for qs in selected.values())
    print(f"Selected {len(selected)} videos:")
    for n_q in sorted(counts.keys(), reverse=True):
        print(f"  {counts[n_q]} videos with {n_q} questions")
    total_q = sum(len(qs) for qs in selected.values())
    print(f"  Total questions: {total_q}")

    return selected


def train_test_split(
    selected: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Split selected videos into train and test sets with a fixed seed."""
    vids = sorted(selected.keys())
    rng = random.Random(GEPA_TRAIN_TEST_SEED)
    rng.shuffle(vids)
    train_vids = vids[:GEPA_TRAIN_SIZE]
    test_vids = vids[GEPA_TRAIN_SIZE:]
    train = {v: selected[v] for v in train_vids}
    test = {v: selected[v] for v in test_vids}
    print(f"\nTrain/test split: {len(train)} train, {len(test)} test videos")
    print(f"  Train questions: {sum(len(qs) for qs in train.values())}")
    print(f"  Test questions: {sum(len(qs) for qs in test.values())}")
    return train, test
