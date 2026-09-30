"""Select stable, jointly observed query-anchor contact segments."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable

import numpy as np

from contactaware.settings import MM


@dataclass(frozen=True)
class ContactTrack:
    start: int
    end: int
    query_ids: np.ndarray
    anchor_path: np.ndarray
    count: int
    score: float


@dataclass(frozen=True)
class StableAnchorEstimate:
    position: np.ndarray
    score: float


def contact_tracks(
    query_ids: np.ndarray,
    phi: np.ndarray,
    surface: np.ndarray,
    *,
    args,
) -> list[ContactTrack]:
    """Return the best non-overlapping set of fixed query-anchor segments."""
    validate_tracking_inputs(query_ids, phi, surface=surface)
    contact = stable_contact_mask(phi, args)
    min_duration = int(args.contact_min_duration)
    candidates = []
    for local_query, query_id in enumerate(query_ids.astype(np.int32)):
        for start, end in active_contact_runs(
            contact[:, local_query], min_duration, max_gap=args.contact_run_max_gap
        ):
            candidates.extend(
                fixed_query_candidates(
                    start,
                    end,
                    int(query_id),
                    surface=surface[:, local_query],
                    args=args,
                )
            )
    return select_nonoverlapping_tracks(
        candidates,
        switch_penalty_frames=max(int(args.contact_switch_penalty_frames), 0),
    )


def fixed_query_candidates(
    start: int,
    end: int,
    query_id: int,
    *,
    surface: np.ndarray,
    args,
) -> list[ContactTrack]:
    def estimate_range(left: int, right: int):
        return stable_anchor(surface[left:right], args)

    min_duration = int(args.contact_min_duration)
    ranges = extract_stable_ranges(
        start,
        end,
        min_duration=min_duration,
        estimate_range=estimate_range,
    )
    tracks = []
    for left, right in ranges:
        estimate = estimate_range(left, right)
        if estimate is None:
            raise ValueError("Stable anchor range lost its estimate")
        # A range ends at its last frame near the anchor: boundary frames sliding away are not this contact.
        # A contact too short to trim is kept whole rather than dropped.
        near = np.flatnonzero(np.linalg.norm(surface[left:right] - estimate.position, axis=1)
                              <= contact_deadband(args))
        trimmed_left, trimmed_right = left + int(near[0]), left + int(near[-1]) + 1
        trimmed = (estimate_range(trimmed_left, trimmed_right)
                   if trimmed_right - trimmed_left >= min_duration else None)
        if trimmed is not None:
            left, right, estimate = trimmed_left, trimmed_right, trimmed
        tracks.append(
            make_fixed_track(
                left,
                right,
                query_id,
                estimate.position,
                score=estimate.score,
            )
        )
    return tracks


def select_nonoverlapping_tracks(
    candidates: list[ContactTrack],
    *,
    switch_penalty_frames: int,
) -> list[ContactTrack]:
    """Maximize covered frames, charging a penalty for every switch between candidate tracks."""
    ordered = sorted(candidates, key=track_signature)
    end_frames = [track.end for track in ordered]
    best: list[tuple[ContactTrack, ...]] = [()]
    for index, track in enumerate(ordered):
        compatible_count = bisect_right(end_frames, track.start, 0, index)
        take = best[compatible_count] + (track,)
        skip = best[index]
        best.append(
            min(
                (skip, take),
                key=lambda items: selection_rank(
                    items,
                    switch_penalty_frames=switch_penalty_frames,
                ),
            )
        )
    return list(best[-1])


def selection_rank(
    tracks: tuple[ContactTrack, ...],
    *,
    switch_penalty_frames: int,
) -> tuple:
    covered_frames = sum(track.count for track in tracks)
    switch_count = max(len(tracks) - 1, 0)
    supported_frames = covered_frames - switch_penalty_frames * switch_count
    fit_cost = sum(track.count * track.score for track in tracks)
    signature = tuple(track_signature(track) for track in tracks)
    return -supported_frames, len(tracks), fit_cost, signature


def track_signature(track: ContactTrack) -> tuple:
    query_id = int(track.query_ids[0])
    anchor = tuple(float(value) for value in track.anchor_path[0])
    return track.end, track.start, query_id, anchor


def extract_stable_ranges(
    start: int,
    end: int,
    *,
    min_duration: int,
    estimate_range: Callable[[int, int], object | None],
) -> list[tuple[int, int]]:
    ranges = []
    cursor = start
    while cursor + min_duration <= end:
        stop = cursor + min_duration
        if estimate_range(cursor, stop) is None:
            cursor += 1
            continue
        while stop < end and estimate_range(cursor, stop + 1) is not None:
            stop += 1
        ranges.append((cursor, stop))
        cursor = stop
    return ranges


def stable_anchor(points: np.ndarray, args) -> StableAnchorEstimate | None:
    distance = pairwise_point_distance(points)
    medoid, inliers = consensus_medoid(distance, contact_deadband(args))
    if float(np.mean(inliers)) < min_stable_fraction(args):
        return None
    return StableAnchorEstimate(
        points[medoid].astype(np.float64),
        distance_score_mm(distance[medoid, inliers], args.contact_score_percentile),
    )


def pairwise_point_distance(points: np.ndarray) -> np.ndarray:
    difference = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(difference, axis=-1)


def consensus_medoid(cost: np.ndarray, threshold: float) -> tuple[int, np.ndarray]:
    if not np.all(np.isfinite(cost)):
        raise ValueError("Stable-contact consensus contains non-finite costs")
    inlier_matrix = cost <= threshold
    counts = np.sum(inlier_matrix, axis=1)
    candidates = np.flatnonzero(counts == np.max(counts))
    inlier_mean = np.asarray(
        [np.mean(cost[index, inlier_matrix[index]]) for index in candidates]
    )
    medoid = int(candidates[int(np.argmin(inlier_mean))])
    return medoid, inlier_matrix[medoid]


def make_fixed_track(
    start: int,
    end: int,
    query_id: int,
    anchor: np.ndarray,
    *,
    score: float,
) -> ContactTrack:
    count = end - start
    return ContactTrack(
        start,
        end,
        np.full(count, query_id, dtype=np.int32),
        np.repeat(np.asarray(anchor, dtype=np.float32)[None], count, axis=0),
        count,
        float(score),
    )


def active_contact_runs(
    mask: np.ndarray,
    min_duration: int,
    *,
    max_gap: int,
) -> tuple[tuple[int, int], ...]:
    if min_duration <= 0:
        raise ValueError(f"contact_min_duration must be positive, got {min_duration}")
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    if active.size == 0:
        return ()
    # Bridge short gaps caused by pose noise.
    split_at = np.flatnonzero(np.diff(active) > int(max_gap) + 1) + 1
    runs = np.split(active, split_at)
    return tuple(
        (int(run[0]), int(run[-1]) + 1)
        for run in runs
        if run.size >= min_duration
    )


def validate_tracking_inputs(
    query_ids: np.ndarray,
    phi: np.ndarray,
    *,
    surface: np.ndarray,
) -> None:
    if phi.ndim != 2:
        raise ValueError(f"Contact phi must be 2-D, got {phi.shape}")
    expected = phi.shape + (3,)
    if surface.shape != expected:
        raise ValueError(
            "Contact scan shapes must be phi=(T,N), surface=(T,N,3); "
            f"got {phi.shape}, {surface.shape}"
        )
    if query_ids.shape != (phi.shape[1],):
        raise ValueError(
            f"query_ids must have shape {(phi.shape[1],)}, got {query_ids.shape}"
        )
    if np.unique(query_ids).size != query_ids.size:
        raise ValueError("Contact query IDs must be unique within one finger group")
    if np.any(query_ids < 0):
        raise IndexError("Contact query IDs must be non-negative")


def contact_deadband(args) -> float:
    deadband = float(args.contact_track_deadband)
    if deadband <= 0.0:
        raise ValueError(f"contact_track_deadband must be positive, got {deadband}")
    return deadband


def min_stable_fraction(args) -> float:
    fraction = float(args.contact_track_min_stable_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "contact_track_min_stable_fraction must be in (0, 1], "
            f"got {fraction}"
        )
    return fraction


def distance_score_mm(distances: np.ndarray, percentile: float) -> float:
    values = np.asarray(distances, dtype=np.float64) * MM
    return float(np.percentile(values, float(percentile)) + np.mean(values))


def stable_contact_mask(phi: np.ndarray, args) -> np.ndarray:
    return (
        (phi < float(args.contact_on_threshold))
        & (phi > -abs(float(args.contact_anchor_max_ref_penetration)))
    )
