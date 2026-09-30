"""Per-sequence wrist mount that keeps a hand trajectory away from Euler gimbal lock.

The floating base rotates through three orthogonal hinges, whose chart is singular where the first and third hinge
axes align. Rotating the frame of the first hinge body (the mount) moves that singularity without changing the hand:
the mounted robot XML differs from the asset only in that body's orientation, and existing trajectories are
re-expressed exactly by refitting the base coordinates to the same palm poses.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from contactaware.solver.pose import _fit_base_pose

MOUNT_FILE = "wrist_mount.json"
ASSET_ROBOT_XML = "asset_robot_xml"
POSITION_TOLERANCE_M = 1e-6
ROTATION_TOLERANCE_RAD = 1e-5
SPHERE_POINTS = 20000
ORIENTATION_ATTRIBUTES = ("quat", "euler", "axisangle", "xyaxes", "zaxis")


def mounted_robot_xml(sequence_dir: Path, hand: str) -> Path | None:
    """The mounted copy of the hand's asset robot in the sequence scene, if the scene has been mounted."""
    mount = Path(sequence_dir) / "scene" / hand / MOUNT_FILE
    if not mount.exists():
        return None
    return mount.parent / json.loads(mount.read_text())[ASSET_ROBOT_XML]


def base_hinges(model: mujoco.MjModel, base_dim: int) -> list[int]:
    joints = [j for j in range(model.njnt)
              if model.jnt_qposadr[j] < base_dim and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if len(joints) != 3:
        raise ValueError(f"Expected three base hinges, found {len(joints)}")
    return joints


def wrist_hinges(model: mujoco.MjModel, base_dim: int) -> tuple[int, int]:
    """First and last hinge joints of the floating base."""
    joints = base_hinges(model, base_dim)
    return joints[0], joints[-1]


def hinge_axes(model: mujoco.MjModel, states: np.ndarray, base_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """World axis of the first hinge (fixed) and of the last hinge per frame."""
    first, last = wrist_hinges(model, base_dim)
    data = mujoco.MjData(model)
    axes = []
    for state in states:
        data.qpos[:len(state)] = state
        mujoco.mj_kinematics(model, data)
        axes.append(data.xaxis[last].copy())
    return data.xaxis[first].copy(), np.asarray(axes)


def clearance(first_axes: np.ndarray, last_axes: np.ndarray) -> np.ndarray:
    """Min over frames of |sin| of the angle between the first and last hinge axes (min |cos| of the middle angle)."""
    return np.sqrt(np.clip(1.0 - (np.atleast_2d(first_axes) @ last_axes.T) ** 2, 0.0, 1.0)).min(axis=1)


def fibonacci_sphere(count: int = SPHERE_POINTS) -> np.ndarray:
    index = np.arange(count) + 0.5
    polar, azimuth = np.arccos(1.0 - 2.0 * index / count), np.pi * (1.0 + 5.0 ** 0.5) * index
    return np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=1)


def best_first_axis(last_axes: np.ndarray, current: np.ndarray) -> np.ndarray:
    """World direction of the first hinge maximizing the clearance; ties keep the direction nearest the current one."""
    candidates = fibonacci_sphere()
    scores = clearance(candidates, last_axes)
    best = np.flatnonzero(scores >= scores.max() - 1e-9)
    return candidates[best[np.argmax(candidates[best] @ current)]]


def mounted_body_quat(model: mujoco.MjModel, base_dim: int, target_axis: np.ndarray) -> tuple[str, np.ndarray]:
    """Orientation of the first hinge body that turns its hinge axis to target_axis by the smallest world rotation."""
    first, _ = wrist_hinges(model, base_dim)
    body = int(model.jnt_bodyid[first])
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    parent = Rotation.from_quat(data.xquat[model.body_parentid[body]][[1, 2, 3, 0]])
    local = Rotation.from_quat(model.body_quat[body][[1, 2, 3, 0]])
    current = (parent * local).apply(model.jnt_axis[first])
    turn, _ = Rotation.align_vectors(target_axis[None], current[None])
    quat = (parent.inv() * turn * parent * local).as_quat()[[3, 0, 1, 2]]
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body), quat / np.linalg.norm(quat)


def write_mounted_robot(source: Path, target: Path, body: str, quat: np.ndarray) -> None:
    """Copy a robot MJCF next to the scene: asset paths re-rooted, the mount body's orientation replaced."""
    text = Path(source).read_text()
    offset = os.path.relpath(Path(source).resolve().parent, Path(target).resolve().parent)
    text = re.sub(r'\bfile="([^"]+)"', lambda m: f'file="{os.path.normpath(os.path.join(offset, m.group(1)))}"', text)
    tag = re.search(rf'<body\b[^>]*\bname="{re.escape(body)}"[^>]*?(/?)>', text)
    if tag is None:
        raise ValueError(f"Body {body} not found in {source}")
    opening = re.sub(rf'\s(?:{"|".join(ORIENTATION_ATTRIBUTES)})="[^"]*"', "", tag.group(0))
    values = " ".join(f"{value:.17g}" for value in quat)
    opening = re.sub(r"\s*(/?)>$", rf' quat="{values}"\1>', opening)
    Path(target).write_text(text[:tag.start()] + opening + text[tag.end():])


def mount_scene(scene_dir: Path, body: str, quat: np.ndarray, report: dict) -> None:
    """Point every scene XML under scene/<hand>/ at mounted copies of the robot files it includes.

    Each included asset robot gets one mounted copy in scene/<hand>/; scenes in subdirectories reference it
    relatively. Copies already in scene/<hand>/ (a mounted scene) have their mount orientation rewritten in place.
    """
    scene_dir = Path(scene_dir).resolve()
    mounted: dict[Path, Path] = {}
    for scene in sorted(scene_dir.rglob("*.xml")):
        text = scene.read_text()
        includes = set(re.findall(r'<include file="([^"]+)"', text))
        for match in includes:
            source = (scene.parent / match).resolve()
            if source not in mounted:
                target = scene_dir / source.name
                if source != target and (target in mounted.values() or target.exists()):
                    raise FileExistsError(f"Mounted robot name collides with {target}")
                write_mounted_robot(source, target, body, quat)
                mounted[source] = target
            text = text.replace(f'<include file="{match}"',
                                f'<include file="{os.path.relpath(mounted[source], scene.parent)}"')
        if includes:
            scene.write_text(text)
    (scene_dir / MOUNT_FILE).write_text(json.dumps({"body": body, "quat": [float(v) for v in quat], **report},
                                                   indent=2) + "\n")


def scene_files(scene_dir: Path) -> list[Path]:
    return [path for path in Path(scene_dir).rglob("*") if path.suffix in (".xml", ".json")]


def scene_snapshot(scene_dir: Path) -> dict[Path, str]:
    """Text of every XML/JSON file under the scene directory, to restore it after a failed mount."""
    return {path: path.read_text() for path in scene_files(scene_dir)}


def restore_scene(scene_dir: Path, snapshot: dict[Path, str]) -> None:
    """Rewrite the snapshot files and delete files created since (e.g. mounted robot copies)."""
    for path in scene_files(scene_dir):
        if path not in snapshot:
            path.unlink()
    for path, text in snapshot.items():
        path.write_text(text)


def hinge_chart(model: mujoco.MjModel, base_dim: int) -> tuple[np.ndarray, str, np.ndarray]:
    """qpos addresses, intrinsic Euler sequence and axis signs of the three co-located orthogonal base hinges."""
    joints = base_hinges(model, base_dim)
    axes = model.jnt_axis[joints]
    order = np.argmax(np.abs(axes), axis=1)
    signs = axes[np.arange(3), order]
    if len(set(order)) != 3 or not np.allclose(axes, np.eye(3)[order] * signs[:, None]):
        raise ValueError("Base hinges must be orthogonal and aligned with body axes")
    return model.jnt_qposadr[joints], "".join("XYZ"[k] for k in order), signs


def refit_states(source_hand, target_hand, states: np.ndarray) -> np.ndarray:
    """Base coordinates of target_hand that reproduce source_hand's palm poses; fingers are unchanged.

    Hinge angles come from an Euler decomposition in the target chart, made continuous in time and shifted by whole
    turns into the joint range; a least-squares refit through forward kinematics then matches the palm pose exactly.
    """
    base = target_hand.base_qpos_dim
    model, data = target_hand.model, target_hand.data
    addresses, sequence, signs = hinge_chart(model, base)
    first_body = int(model.jnt_bodyid[wrist_hinges(model, base)[0]])
    rest = np.array(states[0], dtype=np.float64, copy=True)
    rest[addresses] = 0.0
    target_hand.forward(rest)
    mount = data.xmat[first_body].reshape(3, 3).copy()
    palm_rest = data.xmat[target_hand.palm_body_id].reshape(3, 3).copy()
    poses, angles = [], []
    for state in states:
        source_hand.forward(state)
        position = source_hand.data.xpos[source_hand.palm_body_id].copy()
        rotation = source_hand.data.xmat[source_hand.palm_body_id].reshape(3, 3).copy()
        poses.append((position, rotation))
        chart = mount.T @ rotation @ palm_rest.T @ mount
        angles.append(Rotation.from_matrix(chart).as_euler(sequence) / signs)
    angles = np.unwrap(np.asarray(angles), axis=0)
    lower, upper = model.jnt_range[[np.flatnonzero(model.jnt_qposadr == a)[0] for a in addresses]].T
    turns = np.round(((lower + upper) / 2 - (angles.min(0) + angles.max(0)) / 2) / (2 * np.pi))
    angles += 2 * np.pi * turns
    result = np.array(states, dtype=np.float64, copy=True)
    for t, (position, rotation) in enumerate(poses):
        start = result[t].copy()
        start[addresses] = angles[t]
        result[t] = _fit_base_pose(target_hand, start, position, rotation, 1.0)
    if np.any(result[:, addresses] < lower - 1e-9) or np.any(result[:, addresses] > upper + 1e-9):
        raise ValueError("Refitted wrist angles leave the joint range")
    return result


def hand_bodies(model: mujoco.MjModel, base_dim: int) -> np.ndarray:
    """The rigid hand: the body of the last base hinge and its descendants (virtual wrist links are chart-dependent)."""
    root = int(model.jnt_bodyid[wrist_hinges(model, base_dim)[1]])
    bodies = [root]
    for body in range(root + 1, model.nbody):
        if model.body_parentid[body] in bodies:
            bodies.append(body)
    return np.asarray(bodies)


def pose_error(source_model, target_model, source_states, target_states, base_dim: int) -> tuple[float, float]:
    """Max position (m) and orientation (rad) difference of the hand bodies over all frames."""
    bodies = hand_bodies(source_model, base_dim)
    if not np.array_equal(bodies, hand_bodies(target_model, base_dim)):
        raise ValueError("Mounted robot changed the hand body tree")
    source_data, target_data = mujoco.MjData(source_model), mujoco.MjData(target_model)
    position, orientation = 0.0, 0.0
    for a, b in zip(source_states, target_states):
        source_data.qpos[:len(a)], target_data.qpos[:len(b)] = a, b
        mujoco.mj_kinematics(source_model, source_data)
        mujoco.mj_kinematics(target_model, target_data)
        position = max(position, float(np.abs(source_data.xpos[bodies] - target_data.xpos[bodies]).max()))
        relative = np.einsum("bji,bjk->bik", source_data.xmat[bodies].reshape(-1, 3, 3),
                             target_data.xmat[bodies].reshape(-1, 3, 3))
        orientation = max(orientation, float(Rotation.from_matrix(relative).magnitude().max()))
    return position, orientation


def mount_trajectory(sequence_dir: Path, hand_name: str, source_hand, states: np.ndarray,
                     extra_states: tuple[np.ndarray, ...] = ()) -> tuple[np.ndarray, tuple, dict]:
    """Mount the scene's wrist for a trajectory of source_hand and re-express it (and extra_states) in the new chart.

    The mount maximizes the clearance of `states` from gimbal lock. If verification fails, the scene is restored and
    the inputs are returned unchanged, with the error in the report.
    """
    from contactaware.models.hand_profiles import get_hand_profile
    from contactaware.solver.hand import MujocoHand

    scene_dir = Path(sequence_dir) / "scene" / hand_name
    base = source_hand.base_qpos_dim
    current_axis, last_axes = hinge_axes(source_hand.model, states, base)
    axis = best_first_axis(last_axes, current_axis)
    body, quat = mounted_body_quat(source_hand.model, base, axis)
    report = {ASSET_ROBOT_XML: Path(get_hand_profile(hand_name).mesh_xml).name,
              "clearance_before": float(clearance(current_axis, last_axes)[0]),
              "clearance_chosen": float(clearance(axis, last_axes)[0])}
    snapshot = scene_snapshot(scene_dir)
    try:
        mount_scene(scene_dir, body, quat, report)
        target_hand = MujocoHand(get_hand_profile(hand_name, scene_dir / report[ASSET_ROBOT_XML]), 1.0)
        converted = [refit_states(source_hand, target_hand, np.asarray(x, dtype=np.float64)).astype(np.asarray(x).dtype)
                     for x in (states, *extra_states)]
        position, rotation = pose_error(source_hand.model, target_hand.model, states, converted[0], base)
        if position > POSITION_TOLERANCE_M or rotation > ROTATION_TOLERANCE_RAD:
            raise ValueError(f"Re-expressed trajectory moved the hand: {position:.3g} m, {rotation:.3g} rad")
        report.update(applied=True, clearance_after=float(clearance(*hinge_axes(target_hand.model, converted[0], base))[0]),
                      max_body_position_error_m=position, max_body_rotation_error_rad=rotation)
        (scene_dir / MOUNT_FILE).write_text(json.dumps({**json.loads((scene_dir / MOUNT_FILE).read_text()),
                                                        **report}, indent=2) + "\n")
        return converted[0], tuple(converted[1:]), report
    except Exception as error:
        restore_scene(scene_dir, snapshot)
        return states, tuple(extra_states), {**report, "applied": False, "error": repr(error)}
