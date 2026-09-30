"""Exclude link pairs with negative maximum clearance in the posture search."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from contactaware.cache import CACHE_ROOT as SHARED_CACHE_ROOT

from contactaware.solver.collision import capsule_pair_states

SAMPLES = 2000
SEED = 0
CACHE_ROOT = SHARED_CACHE_ROOT / 'structural_capsules'


def group_clearances(hand, state):
    return capsule_pair_states(hand, *hand.capsule_segments(state))[-1]


def structural_overlaps(hand, *, samples=SAMPLES, seed=SEED):
    rng = np.random.default_rng(seed)
    base = hand.profile.default_qpos.astype(np.float64)
    dim = hand.base_qpos_dim
    lower, upper = hand.lower[dim:], hand.upper[dim:]
    postures = rng.uniform(lower, upper, size=(samples, hand.qpos_dim - dim))
    values = np.asarray([group_clearances(hand, np.r_[base[:dim], p]) for p in postures])
    report = []
    for group in np.flatnonzero(values.max(axis=0) < 0.0):
        start = postures[np.argmax(values[:, group])]
        result = minimize(lambda p: -group_clearances(hand, np.r_[base[:dim], p])[group], start,
                          method='L-BFGS-B', bounds=list(zip(lower, upper)))
        best = -float(result.fun)
        if best < 0.0:
            left, right = hand.capsule_pairs[group]
            report.append(dict(group=int(group), left=hand.capsules[left].geom, right=hand.capsules[right].geom,
                               max_clearance_mm=round(best * 1000.0, 3)))
    return report


def cache_key(hand):
    """Asset, capsule geometry and detection parameters fully determine the result."""
    digest = hashlib.sha256(Path(hand.profile.mesh_xml).read_bytes())
    digest.update(repr((hand.capsules, hand.lower.tolist(), hand.upper.tolist(), hand.profile.default_qpos.tolist(),
                        [tuple(p) for p in hand.capsule_pairs], SAMPLES, SEED)).encode())
    digest.update(Path(__file__).read_bytes())
    return digest.hexdigest()[:32]


def exclude_structural_overlaps(hand):
    path = CACHE_ROOT / f'{hand.profile.name}_{cache_key(hand)}.json'
    if path.exists():
        report = json.loads(path.read_text())
    else:
        report = structural_overlaps(hand)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f'.{os.getpid()}.tmp')  # concurrent cases share a hand: write, then rename
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(path)
    if report:
        keep = np.setdiff1d(np.arange(len(hand.capsule_pairs)), [item['group'] for item in report])
        hand.capsule_pair_groups = hand.capsule_pair_groups[keep]
        hand.capsule_pairs = [hand.capsule_pairs[i] for i in keep]
        hand.capsule_pair_left = hand.capsule_pair_left[keep]
        hand.capsule_pair_right = hand.capsule_pair_right[keep]
    return report
