"""Contact schedules and motion costs shared by all jointly optimized keys."""

import numpy as np

from contactaware.contact.phases import contact_mask_runs
from contactaware.initialization.contact_priority_frame import FIRST_FRAME_PALM_WEIGHT_SCALE
from contactaware.initialization.multi_contact_keys import KeyEvaluation, embed_rows, weighted
from contactaware.solver.temporal import physical_motion_operator, physical_motion_residuals, time_sample_weights
from contactaware.solver.contact_position import contact_position_weights, refined_contact_targets
from contactaware.solver.contact import pad_normal_residuals
from contactaware.solver.contact_tracking import weighted_points
from contactaware.solver.tracking import so3_left_jacobian_inverse, velocity_weights
from contactaware.solver.palm_relative_shape import weighted_shape_terms

def contact_state_weights(mask, frames):
    """Weight each key's force term by the duration of its contact state (mean one)."""
    durations = np.zeros(len(frames), dtype=np.float64)
    for start, end, _ in contact_mask_runs(mask):
        durations[(frames >= start) & (frames <= end)] = end - start + 1
    active = durations > 0
    if not np.any(active):
        return durations
    return durations / durations[active].mean()


def completed_contact_sets(columns):
    sets = tuple(dict.fromkeys(tuple(ids) for ids in columns if len(ids)))
    return tuple(np.asarray(ids, dtype=np.int64) for ids in sets
                 if not any(set(ids) < set(other) for other in sets))


class KeyMotion:
    """Physical key velocity and acceleration using actual elapsed time."""

    def __init__(self, problem, *, fps, seconds, operator_factory=physical_motion_operator,
                 origin_rotation=None):
        self.problem = problem
        times = problem.schedule.frames / fps
        self.durations = time_sample_weights(times, single_frame_seconds=1.0 / fps)
        self.operator = operator_factory(times, seconds, velocity_weights(problem.hand, problem.cfg))
        self.origin_rotation = (problem.hand.palm_pose_jacobian(problem.initial[0])[1]
                                if origin_rotation is None else origin_rotation)

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def motion_terms(self, states):
        residual, jacobian = physical_motion_residuals(self.hand, states, self.operator,
            self.origin_rotation, rotation_jacobian=so3_left_jacobian_inverse,
            coordinate_scale=self.scale)
        return residual, jacobian.toarray()


class ScheduledKeyCosts:
    """Per-key palm, shape, contact-position and pad-normal terms on the trajectory's schedules."""

    def __init__(self, problem, runtime, frame_ids, *, position_multiplier, anchor_costs):
        self.problem, self.runtime, self.frame_ids = problem, runtime, frame_ids
        self.anchor_costs = anchor_costs
        self.blend = np.asarray(runtime.guidance_blend[frame_ids], dtype=np.float64)
        self.participation = np.asarray(runtime.participation[frame_ids], dtype=np.float64)
        # Differentiate the runtime anchor blend through its owner key.
        self.reference_targets = runtime.contact_target_pos_obj[frame_ids]
        self.position_weights = contact_position_weights(problem.cfg, self.participation,
                                                         self.blend, position_multiplier)
        self.normal_weights = (problem.cfg.contact_anchor_weight * problem.cfg.contact_normal_weight_scale
                               * self.participation * self.blend)
        confidence = np.max(self.participation * self.blend, axis=1, initial=0.0)
        # A stable-mask switch does not end the approach/release guidance.
        self.palm_scales = 1.0 - (1.0 - FIRST_FRAME_PALM_WEIGHT_SCALE) * confidence

    def contact_terms(self, index, robot, shared):
        problem, ids = self.problem, self.problem.query_ids
        points, normals, point_rows, normal_rows = shared
        blend = self.blend[index, :, None]
        targets = refined_contact_targets(self.reference_targets[index], self.blend[index],
                                           points - problem.references)
        position_rows = problem.embed(robot.point_jacobians[ids], index) - blend[:, :, None] * point_rows
        normal_residual, facing_rows = pad_normal_residuals(robot.pad_normals[ids],
            problem.embed(robot.pad_jacobians[ids], index), normals,
            problem.hand.query_surface_radii[ids], object_jacobians=normal_rows)
        return [weighted_points(robot.points[ids] - targets, position_rows, self.position_weights[index]),
                weighted_points(normal_residual, facing_rows, self.normal_weights[index])]

    def shape_terms(self, index, state):
        problem = self.problem
        terms = weighted_shape_terms(problem.hand, state, problem.joints_obj[index],
            cfg=problem.cfg, runtime=self.runtime, frame_id=int(self.frame_ids[index]))
        return [(residual, problem.embed(rows, index)) for residual, rows in terms]

    def pose_terms(self, index, state):
        problem = self.problem
        position, rotation, pj, rj = problem.palm_terms(state, problem.joints_obj[index])
        palm_scale = self.palm_scales[index]
        local = (state - problem.initial[index]) / problem.scale
        prior_rows = embed_rows(np.eye(len(local)), index, len(local), problem.dimension)
        return [weighted(position, problem.embed(pj, index), palm_scale * problem.cfg.palm_position_weight),
                weighted(rotation, problem.embed(rj, index), palm_scale * problem.cfg.palm_rotation_weight),
                weighted(local, prior_rows, problem.settings.state_prior_weight)]

    def __call__(self, index, geometry):
        problem, state = self.problem, geometry.states[index]
        terms = self.pose_terms(index, state) + self.shape_terms(index, state)
        terms.extend(self.contact_terms(index, geometry.robots[index], geometry.shared))
        columns = problem.schedule.columns[index]
        if len(columns):
            stable_geometry = tuple(item[columns] for item in geometry.shared)
            terms.extend(self.anchor_costs(index, columns, stable_geometry))
        return terms


class TemporalContactKeys:
    """Shared anchor geometry with one contact schedule and inter-key motion costs."""

    def __init__(self, problem, *, motion, key_costs):
        self.problem, self.motion, self.key_costs = problem, motion, key_costs
        self.cached_vector, self.cached_value = None, None

    def __getattr__(self, name):
        return getattr(self.problem, name)


    def prepare_iteration(self, vector):
        self.problem.prepare_iteration(vector)
        self.cached_vector, self.cached_value = None, None

    def evaluate(self, vector):
        if self.cached_vector is not None and np.array_equal(vector, self.cached_vector):
            return self.cached_value
        geometry = self.constraint_evaluation(vector)
        terms = [(np.sqrt(dt) * residual, np.sqrt(dt) * jacobian)
                 for index, dt in enumerate(self.motion.durations)
                 for residual, jacobian in self.key_costs(index, geometry)]
        terms.append(self.motion.motion_terms(geometry.states))
        value = KeyEvaluation(geometry.states, geometry.shared[0], geometry.shared[1],
            np.concatenate([item[0] for item in terms]), np.vstack([item[1] for item in terms]),
            geometry.diagnostics)
        self.cached_vector, self.cached_value = np.array(vector, copy=True), value
        return value
