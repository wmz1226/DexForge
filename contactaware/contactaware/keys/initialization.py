"""Seed at the contact frame: centered-pad posture solve followed by compact refinement."""


import numpy as np

from contactaware.initialization.centered_pad import centered_contact_query_ids
from contactaware.initialization.key_problem import (
    _centered_seed, _select_initial_problem, _solve_problem,
    key_frame_inputs, RobotAwareKeySettings,
)
from contactaware.initialization.key_state import DEFAULT_CONTACT_AXIAL_CENTER
from contactaware.settings import make_config

GEOMETRIC_SEED_FORCE_WEIGHT = 0.0


def geometric_seed(args, inputs, selection):
    cfg = make_config(args)
    center = DEFAULT_CONTACT_AXIAL_CENTER
    selected = key_frame_inputs(inputs, selection.source_frame, selection.active_anchor_columns)
    query_ids = centered_contact_query_ids(inputs, selection.active_anchor_columns,
        source_frame=selection.source_frame, axial_center=center, cfg=cfg)
    seed, report = _centered_seed(selected, args, query_ids, selected.anchors.anchor_pos_obj[0],
                                  axial_center=center, cfg=cfg)
    return seed.qpos_obj.copy(), report


def refine_geometric_seed(args, inputs, selection):
    state, seed_report = geometric_seed(args, inputs, selection)
    selected = key_frame_inputs(inputs, selection.source_frame, selection.active_anchor_columns)
    query_ids = np.asarray(seed_report['strict_center_query_ids'], dtype=np.int32)
    settings = RobotAwareKeySettings()
    problem, initial_report = _select_initial_problem(selected, state, query_ids,
        selected.anchors.anchor_pos_obj[0], args, settings)
    coordinates, solver_report = _solve_problem(problem, settings)
    final = problem.evaluate(coordinates)
    report = dict(role='Compact initializer; final multi-key physical validation remains mandatory',
        source_frame=selection.source_frame, initialization_feasible=problem.is_feasible(coordinates),
        solver=solver_report, metrics=final.metrics, initial=initial_report, seed=seed_report)
    print('[compact-initializer]', dict(feasible=report['initialization_feasible'],
        violation=problem.constraint_violation(coordinates), status=solver_report['message']), flush=True)
    return final.qpos.copy(), report


def refined_contact_seed(args, inputs, selection):
    """Seed posture without force terms; force closure enters in the joint key problem."""
    state, report = refine_geometric_seed(args, inputs, selection)
    return state, {**report, 'initialization_force_weight': GEOMETRIC_SEED_FORCE_WEIGHT,
                   'joint_force_weight': args.force_closure_weight}

