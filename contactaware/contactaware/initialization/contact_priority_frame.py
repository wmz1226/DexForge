"""Seed posture: oriented pad fitting at the contact frame under collision constraints."""

from dataclasses import dataclass
from functools import lru_cache
import time

import numpy as np
from scipy.optimize import least_squares, minimize

from contactaware.contact.object_model import object_distance_world, obj_to_world
from contactaware.contact.mapping import assign_anchor_queries, make_contact_runtime
from contactaware.settings import BASE_TRANSLATION_DIM, make_config
from contactaware.solver.collision import capsule_pair_states
from contactaware.solver.contact import anchor_outward_normals_world
from contactaware.solver.retarget import FrameProblem
from contactaware.solver.tracking import palm_tracking_linearization
from contactaware.initialization.seeds import exterior_start, prepare_start, starting_postures
from contactaware.initialization.mano_fit import fit_mano_posture

MM_PER_M = 1000.0
CONTACT_ALIGNMENT_MAX_EVALS = 32
FIRST_FRAME_PALM_WEIGHT_SCALE = 0.01
LEAST_SQUARES_GRADIENT_FACTOR = 2.0


def pad_aim_from_points(points, point_jac, targets):
    rays = np.asarray(targets) - points
    lengths = np.linalg.norm(rays, axis=1)
    if np.any(lengths <= np.finfo(np.float64).eps):
        raise ValueError("Contact direction is undefined when a query equals its anchor")
    directions = rays / lengths[:, None]
    projection = np.eye(3)[None] - directions[:, :, None] * directions[:, None, :]
    direction_jac = -np.einsum("nij,njk->nik", projection, point_jac) / lengths[:, None, None]
    return directions, direction_jac


@dataclass(frozen=True)
class GeometryOps:
    distance_world: object
    capsule_states: object
    palm_linearization: object


@dataclass(frozen=True)
class ContactGeometry:
    problem: object
    query_ids: np.ndarray
    targets: np.ndarray
    outward_normals: np.ndarray
    length_scale: float
    ops: GeometryOps
    extra_samples: object = None

    @property
    def evaluation_cache_size(self):
        return 1


    @property
    def state_evaluator(self):
        return self.evaluate

    @property
    def normal_scales(self):
        cfg, hand = self.problem.cfg, self.problem.hand
        return np.sqrt(cfg.contact_normal_weight_scale) * hand.query_surface_radii[self.query_ids] / self.length_scale

    def pad_aim_geometry(self, qpos, query_ids, targets):
        points, point_jac = self.problem.hand.query_positions_jacobian(qpos, query_ids)
        return pad_aim_from_points(points, point_jac, targets)

    def contact_frame_errors(self, points, normals, directions):
        position = (points - self.targets) / self.length_scale
        orientation = self.normal_scales[:, None] * (normals - directions)
        return np.concatenate((position, orientation), axis=1)

    def contact_frame_jacobians(self, point_jac, normal_jac, direction_jac):
        return np.concatenate((point_jac / self.length_scale,
                               self.normal_scales[:, None, None] * (normal_jac - direction_jac)), axis=1)

    def evaluate(self, coordinates):
        qpos = np.asarray(coordinates)
        hand, problem = self.problem.hand, self.problem
        ids = np.arange(len(hand.query_points.local_pos))
        points, jac = hand.query_positions_jacobian(qpos, ids)
        if self.extra_samples is not None:
            extra_points, extra_jac = self.extra_samples.positions_jacobian(hand, qpos)
            points, jac = np.vstack((points, extra_points)), np.concatenate((jac, extra_jac))
        phi, distance_gradient = self.ops.distance_world(points, problem.obj_pose, problem.obj)
        object_jac = np.einsum("ni,nij->nj", distance_gradient, jac)
        self_values, self_jac = self.self_geometry(qpos)
        normals, _ = hand.query_normals_jacobian(qpos, self.query_ids)
        directions, _ = pad_aim_from_points(points[self.query_ids], jac[self.query_ids], self.targets)
        normal_alignment = np.sum(normals * directions, axis=1)
        values = np.concatenate(((phi - problem.cfg.hand_object_safe_distance) / self.length_scale,
                                 self_values / self.length_scale))
        jacobian = np.vstack((object_jac / self.length_scale,
                              self_jac / self.length_scale))
        return {
            "error": (points[self.query_ids] - self.targets) / self.length_scale,
            "contact_frame_error": self.contact_frame_errors(
                points[self.query_ids], normals, directions),
            "constraints": values, "constraint_jac": jacobian,
            "phi": phi, "self_clearance": self_values + problem.cfg.safe_distance,
            "normal_alignment": normal_alignment,
        }

    def self_geometry(self, qpos):
        hand = self.problem.hand
        starts, ends = hand.capsule_segments(qpos)
        left, right, normals, clearances = self.ops.capsule_states(hand, starts, ends)
        left_ids = hand.capsule_body_ids_np[hand.capsule_pair_left]
        right_ids = hand.capsule_body_ids_np[hand.capsule_pair_right]
        delta_jac = hand.point_jacobians(left_ids, left) - hand.point_jacobians(right_ids, right)
        return clearances - self.problem.cfg.safe_distance, np.einsum("ni,nij->nj", normals, delta_jac)

    def palm_residual(self, qpos):
        problem = self.problem
        pos, rot, pos_jac, rot_jac = self.ops.palm_linearization(problem.hand, qpos, problem.joints)
        pos_scale = np.sqrt(problem.cfg.palm_position_weight)
        rot_scale = np.sqrt(problem.cfg.palm_rotation_weight)
        return (np.concatenate((pos_scale * pos, rot_scale * rot)),
                np.vstack((pos_scale * pos_jac, rot_scale * rot_jac)))

    def soft_objective(self, qpos):
        """Unscaled palm cost, used for reports and to break exact ties."""
        problem, hand = self.problem, self.problem.hand
        pos, rot, pos_jac, rot_jac = self.ops.palm_linearization(hand, qpos, problem.joints)
        terms = ((pos, pos_jac, problem.cfg.palm_position_weight),
                 (rot, rot_jac, problem.cfg.palm_rotation_weight))
        cost = sum(weight * float(residual @ residual) for residual, _, weight in terms)
        gradient = sum(2.0 * weight * jac.T @ residual for residual, jac, weight in terms)
        return cost, gradient

    def primary_score(self, qpos):
        # Report the objective in equivalent-distance units.
        residual, _ = self.primary_residual(qpos)
        return self.length_scale * MM_PER_M * float(residual @ residual)


    @property
    def first_stage_palm_scale(self):
        contact_weight = self.problem.cfg.contact_anchor_weight
        if contact_weight <= 0.0:
            raise ValueError("First-frame palm scaling requires positive contact weight")
        return np.sqrt(FIRST_FRAME_PALM_WEIGHT_SCALE / contact_weight) / self.length_scale

    def primary_residual(self, qpos):
        hand = self.problem.hand
        points, point_jac = hand.query_positions_jacobian(qpos, self.query_ids)
        normals, normal_jac = hand.query_normals_jacobian(qpos, self.query_ids)
        directions, direction_jac = pad_aim_from_points(points, point_jac, self.targets)
        residual = self.contact_frame_errors(points, normals, directions).reshape(-1)
        jacobian = self.contact_frame_jacobians(
            point_jac, normal_jac, direction_jac).reshape(-1, len(qpos))
        palm_residual, palm_jacobian = self.palm_residual(qpos)
        palm_scale = self.first_stage_palm_scale
        return (np.concatenate((residual, palm_scale * palm_residual)),
                np.vstack((jacobian, palm_scale * palm_jacobian)))

    def contact_frame_residual(self, qpos):
        """Pad position and normal residuals, used to pre-align the hand before collision constraints apply."""
        qpos = np.asarray(qpos)
        hand, cfg = self.problem.hand, self.problem.cfg
        points, point_jac = hand.query_positions_jacobian(qpos, self.query_ids)
        normals, normal_jac = hand.query_normals_jacobian(qpos, self.query_ids)
        directions, direction_jac = pad_aim_from_points(points, point_jac, self.targets)
        radii = hand.query_surface_radii[self.query_ids, None]
        normal_scale = np.sqrt(cfg.contact_normal_weight_scale)
        palm_residual, palm_jacobian = self.palm_residual(qpos)
        contact_scale = np.sqrt(cfg.contact_anchor_weight)
        palm_residual = palm_residual / contact_scale
        palm_jacobian = palm_jacobian / contact_scale
        residual = np.concatenate(
            ((points - self.targets).ravel(),
             (normal_scale * radii * (normals - directions)).ravel(),
             palm_residual))
        jacobian = np.vstack(
            (point_jac.reshape(-1, hand.qpos_dim),
             (normal_scale * radii[:, :, None] * (normal_jac - direction_jac)).reshape(
                 -1, hand.qpos_dim),
             palm_jacobian))
        return residual / self.length_scale, jacobian / self.length_scale


@dataclass(frozen=True)
class NormalizedProblem:
    geometry: ContactGeometry
    origin: np.ndarray
    scale: np.ndarray
    evaluator: object

    def qpos(self, variables):
        return self.origin + self.scale * variables[:len(self.origin)]

    def values(self, variables):
        return self.evaluator(tuple(self.qpos(variables)))

    def constraints(self, variables, *, contact_cap=None):
        """Collision constraints and per-contact epigraph caps on contact residuals."""
        data = self.values(variables)
        cap = variables[-1] if contact_cap is None else contact_cap
        contact = cap - np.linalg.norm(data["contact_frame_error"], axis=1)
        return np.concatenate((data["constraints"], contact))


    def soft_objective(self, variables):
        cost, gradient = self.geometry.soft_objective(self.qpos(variables))
        return cost, gradient * self.scale

    def collision_constraints(self, variables):
        return self.values(variables)["constraints"]

    def collision_constraint_jacobian(self, variables):
        return self.values(variables)["constraint_jac"] * self.scale[None]

    def primary_objective(self, variables):
        residual, jacobian = self.geometry.primary_residual(self.qpos(variables))
        scaled_jacobian = jacobian * self.scale[None]
        cost = float(residual @ residual)
        gradient = LEAST_SQUARES_GRADIENT_FACTOR * scaled_jacobian.T @ residual
        return cost, gradient


def optimize_primary(problem, *, cfg):
    """Jointly fit pad frames and the reduced-weight palm under collision limits."""
    hand = problem.geometry.problem.hand
    initial = np.zeros(len(problem.origin), dtype=np.float64)
    bounds = list(zip((hand.lower - problem.origin) / problem.scale,
                      (hand.upper - problem.origin) / problem.scale))
    return minimize(problem.primary_objective, initial, jac=True, bounds=bounds,
                    constraints={"type": "ineq", "fun": problem.collision_constraints,
                                 "jac": problem.collision_constraint_jacobian},
                    method="SLSQP",
                    options={"maxiter": cfg.first_frame_max_iterations,
                             "ftol": cfg.first_frame_ftol})


def normalized_problem(geometry, qpos):
    cfg = geometry.problem.cfg
    scale = np.full(len(qpos), cfg.joint_trust_rad)
    scale[:3] = cfg.base_translation_trust_m
    scale[3:geometry.problem.hand.base_qpos_dim] = cfg.base_rotation_trust_rad
    evaluator = lru_cache(maxsize=geometry.evaluation_cache_size)(geometry.state_evaluator)
    return NormalizedProblem(geometry, canonical_base_angles(geometry.problem.hand, qpos), scale, evaluator)


def canonical_base_angles(hand, qpos):
    """Recenter equivalent floating-base hinge angles, preserving physical FK."""
    result = qpos.copy()
    indices = np.arange(BASE_TRANSLATION_DIM, hand.base_qpos_dim)
    period = 2.0 * np.pi
    indices = indices[(hand.upper[indices] - hand.lower[indices]) >= period]
    center = (hand.upper[indices] + hand.lower[indices]) / 2.0
    result[indices] = center + (result[indices] - center + np.pi) % period - np.pi
    return result


def align_contact_frames(geometry, qpos):
    """Unconstrained pre-alignment of all pad frames to their anchors."""
    hand, cfg = geometry.problem.hand, geometry.problem.cfg
    evaluate = lru_cache(maxsize=1)(geometry.contact_frame_residual)
    return least_squares(
        lambda coordinates: evaluate(tuple(coordinates))[0],
        np.asarray(qpos),
        jac=lambda coordinates: evaluate(tuple(coordinates))[1],
        bounds=(hand.lower, hand.upper),
        x_scale="jac",
        ftol=cfg.qp_validation_tolerance,
        xtol=cfg.qp_validation_tolerance,
        gtol=cfg.qp_validation_tolerance,
        max_nfev=CONTACT_ALIGNMENT_MAX_EVALS,
    )


def state_report(geometry, qpos):
    data = geometry.state_evaluator(tuple(qpos))
    errors = np.linalg.norm(data["error"], axis=1) * geometry.length_scale * MM_PER_M
    frame_error = np.max(np.linalg.norm(data["contact_frame_error"], axis=1)) * geometry.length_scale * MM_PER_M
    pos, rot, _, _ = geometry.ops.palm_linearization(geometry.problem.hand, qpos, geometry.problem.joints)
    return {"contact_mm": errors.tolist(), "contact_mean_mm": float(np.mean(errors)),
            "contact_max_mm": float(np.max(errors)),
            "contact_frame_max_mm": float(frame_error),
            "palm_position_error_mm": float(np.linalg.norm(pos)) * MM_PER_M,
            "palm_rotation_error_deg": float(np.degrees(np.linalg.norm(rot))),
            "primary_score_mm": float(geometry.primary_score(qpos)),
            "object_penetration_mm": max(0.0, -float(np.min(data["phi"]))) * MM_PER_M,
            "self_clearance_mm": float(np.min(data["self_clearance"])) * MM_PER_M,
            "normal_alignment_deg": np.degrees(
                np.arccos(np.clip(data["normal_alignment"], -1., 1.))).tolist(),
            "max_normalized_violation": max(0.0, -float(np.min(data["constraints"]))),
            "soft_cost": float(geometry.soft_objective(qpos)[0])}


def result_report(result, geometry, qpos, elapsed):
    return {**state_report(geometry, qpos), "solver_success": bool(result.success),
            "status": int(result.status), "message": str(result.message),
            "iterations": int(result.nit), "seconds": elapsed}


def first_frame_geometry(inputs, args):
    cfg = make_config(args)
    runtime = make_contact_runtime(inputs.anchors, np.arange(len(inputs.obj_pose)),
                                   approach_steps=0, release_steps=0,
                                   mano_vertices=inputs.mano_vertices, obj_pose=inputs.obj_pose)
    runtime = assign_anchor_queries(runtime, inputs.hand, inputs.mano_surface_mapping,
                                    cfg=cfg, contact_query_region=inputs.sequence_policy.contact_query_region)
    columns, ids = stable_frame_mapping(runtime, 0)
    if not len(columns):
        raise ValueError("First-frame contact fitting requires stable contact on frame zero")
    problem = FrameProblem(inputs.hand, cfg, inputs.obj, runtime, inputs.mano_joints[0], inputs.obj_pose[0], 0)
    targets = obj_to_world(runtime.contact_target_pos_obj[0, columns], problem.obj_pose)
    normals = anchor_outward_normals_world(runtime, problem.obj_pose, columns)
    length = float(np.mean(inputs.hand.query_surface_radii[ids]))
    ops = GeometryOps(object_distance_world, capsule_pair_states, palm_tracking_linearization)
    return ContactGeometry(problem, ids, targets, normals, length, ops)


def stable_frame_mapping(runtime, frame_id):
    """Return contacts that are actually stable on this frame."""
    columns = np.flatnonzero(runtime.anchors.mask[frame_id]).astype(np.int32)
    query_ids = runtime.hand_query_ids[frame_id, columns].astype(np.int32)
    if np.any(query_ids < 0):
        invalid = columns[query_ids < 0]
        raise RuntimeError(
            f"Stable anchors have no mapped query at frame {frame_id}: "
            f"{invalid.tolist()}"
        )
    return columns, query_ids


def run_primary_candidates(
    geometry,
    args,
    *,
    initialization_ops,
    query_region,
    candidate_pools=None,
):
    """Rank contact-aligned and exterior seed solves by a shared objective."""
    from contactaware.initialization.first_frame_refinement import (
        full_pad_candidates, nearest_feasible_segment, preserve_noncontact_posture,
        best_correspondence_branch,
        restore_collision_feasibility, restore_self_collision_feasibility, solve_primary,
    )
    problem = geometry.problem
    pools = (
        full_pad_candidates(geometry, query_region)
        if candidate_pools is None
        else tuple(np.asarray(ids, dtype=np.int32) for ids in candidate_pools)
    )
    if len(pools) != len(geometry.query_ids) or any(len(ids) == 0 for ids in pools):
        raise ValueError("Every active contact must have a nonempty correspondence pool")
    posture_seed, fit_report = fit_mano_posture(
        problem.hand, problem.joints, cfg=problem.cfg,
        align_palm=initialization_ops.align_palm)
    print(f"[initialization] current-input-frame MANO fit {fit_report}", flush=True)
    noncontact_reference = posture_seed
    states, reports, primaries = {}, {}, {}

    def solve_candidate(name, primary_start, retreat, route):
        qpos, report = solve_primary(geometry, primary_start)
        report["route"] = route
        primaries[name] = (qpos, primary_start, retreat, report)

    def refine_candidate(name):
        """Search the pad correspondence for one candidate that survived ranking."""
        qpos, primary_start, retreat, report = primaries[name]
        candidate_geometry, qpos, pad_report = best_correspondence_branch(
            geometry, qpos, primary_start, candidates=pools)
        qpos, posture_report = preserve_noncontact_posture(
            candidate_geometry, qpos, noncontact_reference)
        states[name] = (candidate_geometry, qpos)
        reports[name] = {
            **state_report(candidate_geometry, qpos), "initial_solve": report,
            "correspondence_solves": pad_report, "noncontact_posture": posture_report,
            "initial_retreat_mm": retreat,
            "start_contact_max_mm": float(np.max(np.linalg.norm(
                geometry.state_evaluator(tuple(primary_start))["error"], axis=1))
                * geometry.length_scale * MM_PER_M),
        }

    def solve_posture(posture_name, posture):
        start, retreat = prepare_start(posture_name, posture, problem, ops=initialization_ops)
        # Keep pad correspondence fixed during contact-basin selection.
        aligned = align_contact_frames(geometry, start)
        aligned_qpos = np.clip(aligned.x, problem.hand.lower, problem.hand.upper)
        exterior_qpos = start.copy()
        try:
            feasible_aligned_qpos, collision_result = restore_collision_feasibility(
                geometry, aligned_qpos, label=f"{posture_name}_contact")
            collision_report = {"success": True, "message": str(collision_result.message),
                                "converged": bool(collision_result.success),
                                "iterations": int(collision_result.nit)}
        except RuntimeError as error:
            # Use the exterior start when contact-aligned fitting is infeasible.
            tolerance = problem.cfg.geometry_feasibility_tolerance_m / geometry.length_scale
            start_constraints = geometry.state_evaluator(tuple(start))["constraints"]
            self_restoration = None
            if np.min(start_constraints) < -tolerance:
                start, self_result = restore_self_collision_feasibility(
                    geometry, start, label=f"{posture_name}_self")
                self_restoration = {"success": True, "message": str(self_result.message),
                                    "iterations": int(self_result.nit)}
                start_constraints = geometry.state_evaluator(tuple(start))["constraints"]
            if np.min(start_constraints) < -tolerance:
                opened, retreat = exterior_start(problem, start, ops=initialization_ops)
                endpoint_constraints = geometry.state_evaluator(tuple(opened))["constraints"]
                if np.min(endpoint_constraints) < -tolerance:
                    raise RuntimeError(
                        f"No collision-feasible endpoint for {posture_name}: "
                        f"open_seed_violation={max(0.0, -float(np.min(start_constraints))):.3e}, "
                        f"exterior_violation={max(0.0, -float(np.min(endpoint_constraints))):.3e}")
                feasible_endpoint = opened
            else:
                retreat = 0.0
                feasible_endpoint = start
            exterior_qpos = feasible_endpoint.copy()
            feasible_aligned_qpos = nearest_feasible_segment(
                geometry, aligned_qpos, feasible_endpoint)
            collision_report = {"success": False, "message": str(error),
                                "self_restoration": self_restoration,
                                "exterior_retreat_mm": float(retreat),
                                "segment_retreat_mm": float(np.linalg.norm(
                                    feasible_aligned_qpos - aligned_qpos) * MM_PER_M)}
        alignment_route = {
            "route": "contact_aligned_then_feasible",
            "alignment_success": bool(aligned.success), "alignment_status": int(aligned.status),
            "collision_restoration": collision_report,
        }
        solve_candidate(posture_name, feasible_aligned_qpos, retreat, alignment_route)
        solve_candidate(f"{posture_name}_exterior", exterior_qpos, retreat,
                        {"route": "feasible_exterior_start"})

    rejected = {}
    for posture_name, posture in starting_postures(
            problem.hand, posture_seed).items():
        try:
            solve_posture(posture_name, posture)
        except RuntimeError as error:
            # Drop unusable candidates.
            rejected[posture_name] = str(error)
            print(f"[contact-priority] posture {posture_name} rejected: {error}", flush=True)
    if not primaries:
        raise RuntimeError(
            "No starting posture produced a first-frame candidate: "
            + "; ".join(f"{name}: {reason}" for name, reason in rejected.items()))
    # Refine correspondences only for leading candidates.
    from contactaware.initialization.first_frame_refinement import candidate_priority
    ranked = sorted(primaries, key=lambda name: candidate_priority(geometry, primaries[name][0]))
    refined = ranked[:max(1, int(args.first_frame_correspondence_candidates))]
    for name in refined:
        refine_candidate(name)
    for name in ranked[len(refined):]:
        reports[name] = {**state_report(geometry, primaries[name][0]),
                         "initial_solve": primaries[name][3],
                         "correspondence_solves": [],
                         "correspondence_skipped": "outranked in the primary stage",
                         "initial_retreat_mm": primaries[name][2]}
    return states, reports


def summarize_first_frame(report):
    """Keep numerical outcomes without per-vertex indices or duplicate endpoints."""
    refinement = report["refinement"]
    primary = report["primary"]
    solves = [("initial_posture", primary["initial_solve"])]
    solves.extend((f"candidate_pad_{index}", row)
                  for index, row in enumerate(primary["correspondence_solves"]))
    solves.extend((f"pad_{index}", row) for index, row in enumerate(refinement["pad"]))
    fields = ("contact_max_mm", "contact_frame_max_mm", "primary_score_mm", "solver_success", "status", "message",
              "selected", "accepted", "iterations", "seconds", "method", "max_scaled_step")
    return {"selected_initial_state": report["selected_initial_state"],
            "final": report["final"],
            "seconds": report["seconds"], "geometry_setup_seconds": report["geometry_setup_seconds"],
            "original_query_ids": report["original_query_ids"], "final_query_ids": report["final_query_ids"],
            "candidates": {name: compact_candidate(row, fields)
                           for name, row in report["candidates"].items()},
            "solver_stages": [{"stage": name, **compact_solve(row, fields)}
                              for name, row in solves]}


def compact_solve(report, fields):
    return {key: report[key] for key in fields if key in report}


def compact_candidate(report, fields):
    return {**compact_solve(report, fields),
            "initial_solve": compact_solve(report["initial_solve"], fields),
            "correspondence_solves": [compact_solve(solve, fields) for solve in report["correspondence_solves"]]}


def print_first_frame_summary(report):
    summary = summarize_first_frame(report)
    final = summary["final"]
    failures = [f"{row['stage']}:{row['status']}" for row in summary["solver_stages"]
                if row.get("solver_success") is False]
    print(f"[contact-priority] finished seconds={summary['seconds']:.2f} "
          f"contact_mean/max_mm={final['contact_mean_mm']:.3f}/{final['contact_max_mm']:.3f} "
          f"palm_rotation_deg={final['palm_rotation_error_deg']:.2f} "
          f"pad_normal_max_deg={max(final['normal_alignment_deg']):.2f}", flush=True)
    if failures:
        print("[contact-priority] subsolves without convergence (stage:status): "
              + ", ".join(failures), flush=True)


def solve_first_frame(args, inputs, *, initialization_ops, geometry_transform):
    """Solve the guided contact frame with fixed pad-center queries under object-mesh and self-collision constraints."""
    from contactaware.initialization.centered_pad import fixed_query_pools, refine_fixed_queries
    from contactaware.initialization.first_frame_refinement import candidate_priority
    geometry = first_frame_geometry(inputs, args)
    original_query_ids = geometry.query_ids.copy()
    setup_started = time.perf_counter()
    geometry = geometry_transform(geometry)
    setup_seconds = time.perf_counter() - setup_started
    print(f"[contact-priority] start targets={len(geometry.targets)} objective=paired_pad_and_palm", flush=True)
    started = time.perf_counter()
    query_region = inputs.sequence_policy.contact_query_region
    states, reports = run_primary_candidates(geometry, args, initialization_ops=initialization_ops,
                                             query_region=query_region, candidate_pools=fixed_query_pools(geometry))
    selected = min(states, key=lambda name: candidate_priority(*states[name]))
    geometry, primary = states[selected]
    final_geometry, final, refinement = refine_fixed_queries(geometry, primary)
    report = {"selected_initial_state": selected, "primary": reports[selected], "candidates": reports,
              "final": state_report(final_geometry, final), "refinement": refinement,
              "seconds": time.perf_counter() - started, "geometry_setup_seconds": setup_seconds,
              "original_query_ids": original_query_ids.tolist(),
              "final_query_ids": final_geometry.query_ids.tolist()}
    print_first_frame_summary(report)
    return final, final_geometry.query_ids, report
