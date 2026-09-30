"""Stable MANO contact anchors on the object surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contactaware.settings import FINGER_LABELS
from contactaware.types import ContactAnchorSet, ObjectModel
from contactaware.contact.object_model import object_query_obj, world_to_obj
from contactaware.contact.stable import ContactTrack, contact_tracks

POSITION_DECIMALS = 6
SURFACE_MATCH_TOLERANCE_M = 1.0e-6
NORMAL_EPS = 1.0e-8


@dataclass(frozen=True)
class ManoContactEpisode:
    finger_id: int
    track: ContactTrack


def make_mano_contact_anchors(
    vertices: np.ndarray,
    obj_pose: np.ndarray,
    vertex_fingers: np.ndarray,
    *,
    obj: ObjectModel,
    args,
) -> ContactAnchorSet:
    allowed = np.flatnonzero(vertex_fingers >= 0)
    phi, surface, query_obj = mano_contact_scan(vertices[:, allowed], obj_pose, obj)
    episodes = []
    for finger_id in sorted(set(vertex_fingers[allowed].astype(int))):
        local = np.flatnonzero(vertex_fingers[allowed] == finger_id)
        tracks = contact_tracks(
            allowed[local],
            phi[:, local],
            surface[:, local],
            args=args,
        )
        episodes.extend(ManoContactEpisode(int(finger_id), track) for track in tracks)
    log_mano_contact_episodes(episodes)
    return materialize_mano_anchors(
        vertices.shape[0],
        episodes,
        allowed_vertex_ids=allowed,
        vertex_fingers=vertex_fingers,
        phi=phi,
        surface=surface,
        query_obj=query_obj,
    )


def mano_contact_scan(
    vertices: np.ndarray,
    obj_pose: np.ndarray,
    obj: ObjectModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi_rows, surface_rows, query_rows = [], [], []
    for frame_id in range(vertices.shape[0]):
        points_obj = world_to_obj(vertices[frame_id], obj_pose[frame_id])
        phi, surface, _ = object_query_obj(points_obj, obj)
        phi_rows.append(phi)
        surface_rows.append(surface)
        query_rows.append(points_obj)
    return tuple(np.asarray(rows) for rows in (phi_rows, surface_rows, query_rows))


def log_mano_contact_episodes(episodes: list[ManoContactEpisode]) -> None:
    print(f"  MANO denoised contact tracks: {len(episodes)}")
    for episode in episodes:
        track = episode.track
        label = FINGER_LABELS[int(episode.finger_id)]
        print(
            f"    {label}: frames {track.start}-{track.end - 1} "
            f"len={track.count} vertex={int(track.query_ids[0])} "
            f"score={track.score:.3f}mm"
        )


def materialize_mano_anchors(
    frame_count: int,
    episodes: list[ManoContactEpisode],
    *,
    allowed_vertex_ids: np.ndarray,
    vertex_fingers: np.ndarray,
    phi: np.ndarray,
    surface: np.ndarray,
    query_obj: np.ndarray,
    surface_normal: np.ndarray | None = None,
) -> ContactAnchorSet:
    if not episodes:
        raise RuntimeError("No reliable MANO contact tracks were found")
    if surface_normal is not None and np.shape(surface_normal) != np.shape(surface):
        raise ValueError("Surface normals must match the surface-point array")
    episode_count = len(episodes)
    mask = np.zeros((frame_count, episode_count), dtype=bool)
    anchors = np.zeros((frame_count, episode_count, 3), dtype=np.float32)
    normals = np.zeros_like(anchors)
    fingers = np.zeros(episode_count, dtype=np.int32)
    vertices = np.full((frame_count, episode_count), -1, dtype=np.int32)
    for column, episode in enumerate(episodes):
        write_mano_episode(
            mask, anchors, vertices, episode=episode, column=column
        )
        active = np.flatnonzero(mask[:, column])
        normals[active, column] = contact_anchor_outward_normal(
            episode,
            allowed_vertex_ids=allowed_vertex_ids,
            vertex_fingers=vertex_fingers,
            phi=phi,
            surface=surface,
            query_obj=query_obj,
            surface_normal=surface_normal,
        )
        fingers[column] = int(episode.finger_id)
    summary = contact_track_summary(
        mask, anchors, vertices, finger_ids=fingers
    )
    return ContactAnchorSet(mask, anchors, normals, fingers, vertices, summary)


def contact_anchor_outward_normal(
    episode: ManoContactEpisode,
    *,
    allowed_vertex_ids: np.ndarray,
    vertex_fingers: np.ndarray,
    phi: np.ndarray,
    surface: np.ndarray,
    query_obj: np.ndarray,
    surface_normal: np.ndarray | None = None,
) -> np.ndarray:
    track = episode.track
    frames = np.arange(track.start, track.end, dtype=np.int64)
    local_ids = np.flatnonzero(
        vertex_fingers[allowed_vertex_ids] == int(episode.finger_id)
    )
    candidates = surface[frames[:, None], local_ids[None, :]]
    distance = np.linalg.norm(candidates - track.anchor_path[0], axis=-1)
    frame_offset, finger_offset = np.unravel_index(np.argmin(distance), distance.shape)
    if distance[frame_offset, finger_offset] > SURFACE_MATCH_TOLERANCE_M:
        raise ValueError("Stable anchor does not match a source surface sample")
    frame = int(frames[frame_offset])
    local_id = int(local_ids[finger_offset])
    if surface_normal is not None:
        return normalized_surface_normal(surface_normal[frame, local_id])
    direction = query_obj[frame, local_id] - surface[frame, local_id]
    if phi[frame, local_id] < 0.0:
        direction = -direction
    length = float(np.linalg.norm(direction))
    if not np.all(np.isfinite(direction)) or length <= NORMAL_EPS:
        raise ValueError("Stable anchor has an invalid object outward normal")
    return direction / length


def normalized_surface_normal(normal: np.ndarray) -> np.ndarray:
    direction = np.asarray(normal, dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or length <= NORMAL_EPS:
        raise ValueError("Stable anchor has an invalid surface normal")
    return direction / length


def write_mano_episode(
    mask: np.ndarray,
    anchors: np.ndarray,
    vertices: np.ndarray,
    *,
    episode: ManoContactEpisode,
    column: int,
) -> None:
    track = episode.track
    frames = np.arange(track.start, track.end, dtype=np.int64)
    if track.anchor_path.shape != (len(frames), 3):
        raise ValueError("Contact anchor path length does not match its active frames")
    if track.query_ids.shape != (len(frames),):
        raise ValueError("Contact query path length does not match its active frames")
    mask[frames, column] = True
    anchors[frames, column] = track.anchor_path
    vertices[frames, column] = track.query_ids


def contact_track_summary(
    mask: np.ndarray,
    anchors: np.ndarray,
    query_ids: np.ndarray,
    *,
    finger_ids: np.ndarray,
) -> dict:
    segments = [
        mano_contact_segment(
            column,
            mask=mask,
            anchors=anchors,
            query_ids=query_ids,
            finger_ids=finger_ids,
        )
        for column in range(mask.shape[1])
    ]
    return {
        "segment_count": len(segments),
        "segments": segments,
    }


def mano_contact_segment(
    column: int,
    *,
    mask: np.ndarray,
    anchors: np.ndarray,
    query_ids: np.ndarray,
    finger_ids: np.ndarray,
) -> dict:
    frames = np.flatnonzero(mask[:, column])
    if frames.size == 0 or not np.array_equal(
        frames, np.arange(frames[0], frames[-1] + 1)
    ):
        raise ValueError(f"MANO contact segment {column} is empty or non-contiguous")
    anchor_path = anchors[frames, column]
    if not np.allclose(anchor_path, anchor_path[0], atol=1e-7):
        raise ValueError(f"MANO contact segment {column} has a moving anchor")
    stable_queries = np.unique(query_ids[frames, column])
    if stable_queries.size != 1:
        raise ValueError(f"MANO contact segment {column} has a moving query")
    finger_id = int(finger_ids[column])
    return {
        "finger": FINGER_LABELS[finger_id],
        "start_frame": int(frames[0]),
        "end_frame": int(frames[-1]),
        "frame_count": int(frames.size),
        "anchor_pos_obj": np.round(
            anchor_path[0].astype(float), POSITION_DECIMALS
        ).tolist(),
        "mano_vertex_id": int(stable_queries[0]),
    }
