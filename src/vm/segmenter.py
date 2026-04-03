from dataclasses import dataclass


@dataclass
class Segment:
    index: int
    start_s: int
    end_s: int


def segment_video(duration_s: float, segment_length_s: int = 30) -> list[Segment]:
    segments = []
    start = 0
    idx = 0
    end_s = int(duration_s)
    while start < end_s:
        end = min(start + segment_length_s, end_s)
        if end > start:
            segments.append(Segment(index=idx, start_s=start, end_s=end))
            idx += 1
        start += segment_length_s
    return segments
