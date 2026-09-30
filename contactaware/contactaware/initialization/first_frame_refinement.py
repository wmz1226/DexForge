"""Seed refinement: primary solve, collision restoration and pad-correspondence search."""

from dataclasses import replace
import time

import numpy as np
from contactaware.solver.restoration import minimize_restoration

from contactaware.contact.mapping import contact_query_ids
from contactaware.initialization.contact_priority_frame import (
    MM_PER_M, normalized_problem, optimize_primary, result_report,
)

TERMINAL_TIP_AXIAL = 1.0
PALMAR_NORMAL_COMPONENT = 4
JACOBIAN_ACTIVITY_EPS = 1e-10
NONCONTACT_REFERENCE_FACTORS = (0.5, 0.25, 0.125, 0.0625, 0.03125)


def candidate_priority(geometry, qpos):
    """Rank by feasibility, then the pad-frame and reduced-palm cost; the unscaled palm cost breaks ties."""
    data = geometry.state_evaluator(tuple(qpos))
    tolerance = geometry.problem.cfg.geometry_feasibility_tolerance_m / geometry.length_scale
    violation = max(0.0, -float(np.min(data["constraints"])) - tolerance)
    residual, _ = geometry.primary_residual(qpos)
    return violation, float(residual @ residual), float(geometry.soft_objective(qpos)[0])


def not_worse(candidate, incumbent, slack=1e-9):
    """Compare two :func:`candidate_priority` tuples with slack on the soft terms."""
    return candidate <= (incumbent[0], incumbent[1] + slack, incumbent[2] + slack)


def solve_primary(geometry, qpos):
    """Jointly fit pad frames and the 0.01-weight palm under collision limits."""
    problem = normalized_problem(geometry, qpos)
    started = time.perf_counter()
    result = optimize_primary(problem, cfg=geometry.problem.cfg)
    result.method = "joint_pad_and_reduced_weight_palm_pose_least_squares"
    result.max_scaled_step = float(np.max(np.abs(result.x[:len(problem.origin)])))
    hand = geometry.problem.hand
    endpoint = np.clip(problem.qpos(result.x), hand.lower, hand.upper)
    initial = np.clip(np.asarray(qpos), hand.lower, hand.upper)
    selected = "solver_endpoint"
    final = endpoint
    if candidate_priority(geometry, initial) <= candidate_priority(geometry, endpoint):
        selected = "input_iterate"
        final = initial
    report = result_report(result, geometry, final, time.perf_counter() - started)
    report.update(selected=selected, method=getattr(result, "method", "SLSQP"),
                  max_scaled_step=float(getattr(result, "max_scaled_step", 0.0)))
    return final, report


def preserve_noncontact_posture(geometry, qpos, reference):
    """Move non-contact joints toward the MANO fit by a feasible interpolation that keeps contact quality."""
    hand = geometry.problem.hand
    current = np.asarray(qpos, dtype=np.float64).copy()
    reference = np.asarray(reference, dtype=np.float64)
    _, jacobian = hand.query_positions_jacobian(current, geometry.query_ids)
    active = np.flatnonzero(np.linalg.norm(jacobian, axis=(0, 1)) > JACOBIAN_ACTIVITY_EPS)
    active = np.union1d(active, np.arange(hand.base_qpos_dim, dtype=np.int32))
    inactive = np.setdiff1d(np.arange(hand.qpos_dim, dtype=np.int32), active)
    if not len(inactive):
        return current, {"inactive_joints": [], "method": "none"}
    current_priority = candidate_priority(geometry, current)
    slack = geometry.problem.cfg.geometry_feasibility_tolerance_m / geometry.length_scale
    target = current.copy()
    target[inactive] = np.clip(reference[inactive], hand.lower[inactive], hand.upper[inactive])
    target_constraints = geometry.state_evaluator(tuple(target))["constraints"]
    if (np.min(target_constraints) >= -slack
            and not_worse(candidate_priority(geometry, target), current_priority)):
        return target, {"inactive_joints": inactive.tolist(), "method": "direct_reference"}
    for factor in NONCONTACT_REFERENCE_FACTORS:
        candidate = current.copy()
        candidate[inactive] = current[inactive] + factor * (target[inactive] - current[inactive])
        constraints = geometry.state_evaluator(tuple(candidate))["constraints"]
        if (np.min(constraints) >= -slack
                and not_worse(candidate_priority(geometry, candidate), current_priority)):
            return candidate, {"inactive_joints": inactive.tolist(),
                               "method": "feasible_reference_interpolation", "factor": factor}
    return current, {"inactive_joints": inactive.tolist(), "method": "reference_infeasible"}


def restore_collision_feasibility(geometry, qpos, *, label="feasibility"):
    """Remove collision penetration while staying closest to a contact-fit pose."""
    hand, cfg = geometry.problem.hand, geometry.problem.cfg
    origin = np.asarray(qpos, dtype=np.float64).copy()
    scale = np.full(hand.qpos_dim, cfg.joint_trust_rad, dtype=np.float64)
    scale[:3] = cfg.base_translation_trust_m
    scale[3:hand.base_qpos_dim] = cfg.base_rotation_trust_rad

    def objective(coordinates):
        delta = (np.asarray(coordinates) - origin) / scale
        return 0.5 * float(delta @ delta), delta / scale

    def constraints(coordinates):
        return geometry.state_evaluator(tuple(coordinates))["constraints"]

    def constraint_jacobian(coordinates):
        return geometry.state_evaluator(tuple(coordinates))["constraint_jac"]

    result = minimize_restoration(
        objective,
        origin,
        jac=True,
        bounds=list(zip(hand.lower, hand.upper)),
        constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
        method="SLSQP",
        options={"maxiter": cfg.qp_maxiter, "ftol": cfg.qp_validation_tolerance},
        patience=cfg.sqp_stagnation_patience,
        relative_tolerance=cfg.sqp_progress_relative_tolerance,
        feasibility_tolerance=cfg.geometry_feasibility_tolerance_m / geometry.length_scale,
    )
    if not result.feasible:
        raise RuntimeError(
            f"Collision restoration failed for {label}: {result.message}; "
            f"max_violation={max(0.0, -float(np.min(constraints(result.x)))):.3e}"
        )
    restored = np.clip(result.x, hand.lower, hand.upper)
    return restored, result


def restore_self_collision_feasibility(geometry, qpos, *, label="self_feasibility"):
    """Find the closest pose satisfying the hand's capsule clearances."""
    hand, cfg = geometry.problem.hand, geometry.problem.cfg
    origin = np.asarray(qpos, dtype=np.float64).copy()
    scale = np.full(hand.qpos_dim, cfg.joint_trust_rad, dtype=np.float64)
    scale[:3] = cfg.base_translation_trust_m
    scale[3:hand.base_qpos_dim] = cfg.base_rotation_trust_rad

    def objective(coordinates):
        delta = (np.asarray(coordinates) - origin) / scale
        return 0.5 * float(delta @ delta), delta / scale

    def constraints(coordinates):
        return geometry.self_geometry(coordinates)[0] / geometry.length_scale

    def constraint_jacobian(coordinates):
        return geometry.self_geometry(coordinates)[1] / geometry.length_scale

    result = minimize_restoration(
        objective,
        origin,
        jac=True,
        bounds=list(zip(hand.lower, hand.upper)),
        constraints={"type": "ineq", "fun": constraints, "jac": constraint_jacobian},
        method="SLSQP",
        options={"maxiter": cfg.qp_maxiter, "ftol": cfg.qp_validation_tolerance},
        patience=cfg.sqp_stagnation_patience,
        relative_tolerance=cfg.sqp_progress_relative_tolerance,
        feasibility_tolerance=cfg.geometry_feasibility_tolerance_m / geometry.length_scale,
    )
    # Match the constraint length scaling.
    violation = max(0.0, -float(np.min(constraints(result.x))))
    tolerance = cfg.geometry_feasibility_tolerance_m / geometry.length_scale
    if not result.feasible or violation > tolerance:
        raise RuntimeError(
            f"Self-collision restoration failed for {label}: {result.message}; "
            f"max_violation={violation:.3e}"
        )
    restored = np.clip(result.x, hand.lower, hand.upper)
    return restored, result


def nearest_feasible_segment(geometry, contact_qpos, exterior_qpos):
    """Return the closest collision-feasible point on the contact-to-exterior path."""
    cfg = geometry.problem.cfg
    start = np.asarray(contact_qpos, dtype=np.float64)
    end = np.asarray(exterior_qpos, dtype=np.float64)
    direction = end - start
    if np.linalg.norm(direction) <= np.finfo(np.float64).eps:
        return end.copy()

    # Match the constraint length scaling.
    slack = cfg.geometry_feasibility_tolerance_m / geometry.length_scale

    def feasible(fraction):
        pose = start + fraction * direction
        return float(np.min(geometry.state_evaluator(tuple(pose))["constraints"])) >= -slack

    if feasible(0.0):
        return start.copy()
    if not feasible(1.0):
        violation = -float(np.min(geometry.state_evaluator(tuple(end))["constraints"]))
        raise RuntimeError(
            f"Exterior endpoint is not collision-feasible: violation={violation:.3e}"
            f" (slack {slack:.3e})")
    low, high = 0.0, 1.0
    tolerance = cfg.qp_validation_tolerance / np.linalg.norm(direction)
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if feasible(middle):
            high = middle
        else:
            low = middle
    return start + high * direction


def full_pad_candidates(geometry, query_region):
    hand, cfg = geometry.problem.hand, geometry.problem.cfg
    full_pad_cfg = replace(cfg, pad_max_axial=TERMINAL_TIP_AXIAL)
    candidates = []
    for query_id in geometry.query_ids:
        finger = int(hand.query_points.finger_ids[query_id])
        ids = contact_query_ids(hand, finger, query_region, cfg=full_pad_cfg)
        # Require a palmar position and normal.
        normals = hand.query_surface_descriptors.values[ids, PALMAR_NORMAL_COMPONENT]
        ids = ids[normals >= cfg.pad_min_palmar]
        if not len(ids):
            raise ValueError(f"Finger {finger} has no outward-facing palmar queries")
        candidates.append(ids)
    return tuple(candidates)


def nearest_pad_geometry(geometry, qpos, candidates):
    ids = np.array([oriented_pad_match(geometry, qpos, pool, target)
                    for pool, target in zip(candidates, geometry.targets)],
                   dtype=np.int32)
    return replace(geometry, query_ids=ids)


def oriented_pad_match(geometry, qpos, pool, target):
    hand, cfg = geometry.problem.hand, geometry.problem.cfg
    points = hand.query_positions(qpos, pool)
    normals, _ = hand.query_normals_jacobian(qpos, pool)
    targets = np.broadcast_to(target, points.shape)
    directions, _ = geometry.pad_aim_geometry(qpos, pool, targets)
    normal_error = hand.query_surface_radii[pool, None] * (normals - directions)
    cost = np.sum((points - target)**2, axis=1) / geometry.length_scale**2
    cost += cfg.contact_normal_weight_scale * np.sum(normal_error**2, axis=1) / geometry.length_scale**2
    return pool[np.argmin(cost)]


def refine_pad_correspondence(geometry, qpos, *, candidates):
    current, pose = geometry, qpos.copy()
    reports = []
    while True:
        proposal = nearest_pad_geometry(current, pose, candidates)
        if np.array_equal(proposal.query_ids, current.query_ids):
            return current, pose, reports
        candidate, report = solve_primary(proposal, pose)
        accepted = candidate_priority(proposal, candidate) < candidate_priority(current, pose)
        reports.append({**report, "accepted": accepted})
        if not accepted:
            return current, pose, reports
        current, pose = proposal, candidate


def best_correspondence_branch(geometry, solved_qpos, seed_qpos, *, candidates):
    """Refine correspondences from the primary solution and from its seed; keep the better branch."""
    solved = np.asarray(solved_qpos, dtype=np.float64)
    seed = np.asarray(seed_qpos, dtype=np.float64)
    starts = [("solved", solved)]
    if not np.array_equal(solved, seed):
        starts.append(("seed", seed))
    branches = []
    for label, start in starts:
        branch_geometry, pose, reports = refine_pad_correspondence(geometry, start, candidates=candidates)
        branches.append((candidate_priority(branch_geometry, pose), label,
                         branch_geometry, pose, reports))
    _, label, branch_geometry, pose, reports = min(branches, key=lambda row: row[0])
    annotated = [{**report, "branch": label} for report in reports]
    annotated.append({"branch_selection": label, "branch_priority": {
        row[1]: {"violation": row[0][0], "worst_contact_frame_mm":
                 row[0][1] * branch_geometry.length_scale * MM_PER_M, "palm_cost": row[0][2]}
        for row in branches}})
    return branch_geometry, pose, annotated

