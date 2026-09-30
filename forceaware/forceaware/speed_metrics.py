from __future__ import annotations


def optimization_fps(*, frame_count: int, optimization_time_sec: float) -> float:
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if optimization_time_sec <= 0.0:
        raise ValueError(
            f"optimization_time_sec must be positive, got {optimization_time_sec:g}"
        )
    return float(frame_count) / float(optimization_time_sec)
