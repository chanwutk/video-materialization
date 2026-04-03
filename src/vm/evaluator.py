def evaluate(predictions: list[dict]) -> dict:
    """
    predictions: list of {video_id, question_key, predicted_id, answer_id, policy}

    Returns: {accuracy, correct, total, unparsed, per_video: {video_id: {correct, total, accuracy}}}
    """
    correct = 0
    total = 0
    unparsed = 0
    per_video: dict[str, dict] = {}

    for pred in predictions:
        vid = pred["video_id"]
        if vid not in per_video:
            per_video[vid] = {"correct": 0, "total": 0}

        total += 1
        per_video[vid]["total"] += 1

        if pred["predicted_id"] is None:
            unparsed += 1
            continue

        if pred["predicted_id"] == pred["answer_id"]:
            correct += 1
            per_video[vid]["correct"] += 1

    for vid_stats in per_video.values():
        vid_stats["accuracy"] = (
            vid_stats["correct"] / vid_stats["total"] if vid_stats["total"] > 0 else 0.0
        )

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "unparsed": unparsed,
        "per_video": per_video,
    }
