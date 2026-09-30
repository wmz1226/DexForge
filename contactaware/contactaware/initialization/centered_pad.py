"""Fixed axial-center contacts on the full terminal surface."""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from contactaware.contact.mano_surface import mano_reference_descriptor
from contactaware.contact.mapping import contact_query_ids
from contactaware.contact.surface import (
    AXIAL_COMPONENT,
    SurfaceDescriptorSet,
    subset_descriptors,
    contact_surface_descriptor_cost,
)
from contactaware.initialization.first_frame_refinement import (
    restore_collision_feasibility,
    solve_primary,
)
from contactaware.initialization.first_frame_speed import accelerated_geometry
from contactaware.types import FULL_TERMINAL_SURFACE, RetargetInputs


def centered_contact_query_ids(
    inputs: RetargetInputs,
    anchor_columns: np.ndarray,
    *,
    source_frame: int,
    axial_center: float,
    cfg,
) -> np.ndarray:
    """Match each anchor's anatomical region at the axial-center reference."""
    if not 0.0 <= axial_center <= 1.0:
        raise ValueError("Contact axial center must lie in [0, 1]")
    query_ids = [
        centered_anchor_query(
            inputs,
            int(column),
            source_frame=source_frame,
            axial_center=axial_center,
            cfg=cfg,
        )
        for column in np.asarray(anchor_columns, dtype=np.int32)
    ]
    return np.asarray(query_ids, dtype=np.int32)


def centered_anchor_query(
    inputs: RetargetInputs,
    column: int,
    *,
    source_frame: int,
    axial_center: float,
    cfg,
) -> int:
    finger_id = int(inputs.anchors.finger_ids[column])
    vertex_id = int(inputs.anchors.mano_vertex_ids[source_frame, column])
    reference = mano_reference_descriptor(
        inputs.mano_surface_mapping,
        vertex_id=vertex_id,
        finger_id=finger_id,
        merged_terminal=finger_id in inputs.hand.profile.merged_terminal_finger_ids,
    )
    candidate_ids = contact_query_ids(
        inputs.hand,
        finger_id,
        query_region=FULL_TERMINAL_SURFACE,
        cfg=replace(cfg, pad_min_axial=0.0, pad_max_axial=1.0),
    )
    candidates = subset_descriptors(
        inputs.hand.query_surface_descriptors,
        candidate_ids,
    )
    # Center axially while preserving azimuth and normal.
    values = reference.values.copy()
    values[:, AXIAL_COMPONENT] = axial_center
    centered = SurfaceDescriptorSet(values, reference.azimuth_valid.copy())
    cost = contact_surface_descriptor_cost(
        centered,
        candidates,
        minimum_palmar=cfg.pad_min_palmar,
        axial_weight=cfg.descriptor_axial_cost_weight,
        azimuth_weight=cfg.descriptor_azimuth_cost_weight,
    )
    if not np.any(np.isfinite(cost)):
        raise ValueError(f"No finite center-surface query for finger_id={finger_id}")
    selected = int(np.argmin(cost))
    return int(candidate_ids[selected])


def centered_geometry_transform(
    key_inputs: RetargetInputs,
    anchor_columns: np.ndarray,
    *,
    axial_center: float,
    cfg,
):
    """Return fixed center queries and a transform applied before pose solving."""
    query_ids = centered_contact_query_ids(
        key_inputs,
        anchor_columns,
        source_frame=0,
        axial_center=axial_center,
        cfg=cfg,
    )
    def transform(geometry):
        if query_ids.shape != geometry.query_ids.shape:
            raise ValueError("Centered pad queries do not match active contact targets")
        expected = key_inputs.hand.query_points.finger_ids[geometry.query_ids]
        actual = key_inputs.hand.query_points.finger_ids[query_ids]
        if not np.array_equal(actual, expected):
            raise ValueError("Centered pad mapping changed contact-to-finger identity")
        return replace(accelerated_geometry(geometry), query_ids=query_ids)

    return query_ids, transform


def fixed_query_pools(geometry):
    """Keep the selected material query fixed throughout contact refinement."""
    return tuple(np.asarray([query_id], dtype=np.int32) for query_id in geometry.query_ids)


def refine_fixed_queries(geometry, qpos):
    """Audit active collision queries while retaining fixed pad-center identities."""
    final_geometry, final, solves = geometry, np.asarray(qpos, dtype=np.float64), []
    while True:
        violated = final_geometry.full_collision_violations(final)
        if final_geometry.collision_query_ids is None:
            # None already constrains every query.
            added = np.empty(0, dtype=np.int32)
        else:
            added = np.setdiff1d(violated, final_geometry.collision_query_ids)
        if not len(added):
            break
        final_geometry = final_geometry.add_collision_queries(added)
        # Add violated queries to the primary solve.
        try:
            restored, restoration = restore_collision_feasibility(
                final_geometry,
                final,
                label=f"fixed_center_collision_{len(solves):02d}",
            )
            restoration_iterations, restoration_message = int(restoration.nit), None
        except RuntimeError as error:
            restored, restoration_iterations = final, None
            restoration_message = str(error)
        final, solve = solve_primary(final_geometry, restored)
        tolerance = (
            final_geometry.problem.cfg.geometry_feasibility_tolerance_m
            / final_geometry.length_scale
        )
        constraint_values = final_geometry.state_evaluator(tuple(final))["constraints"]
        violation = max(0.0, -float(np.min(constraint_values)))
        print(f"  [fixed-center] pass={len(solves)} added={len(added)} "
              f"violation={violation:.3e} "
              f"restoration={'ok' if restoration_message is None else 'failed'}", flush=True)
        if violation > tolerance:
            raise RuntimeError("Fixed-center collision refinement returned an infeasible state")
        solves.append(
            {
                **solve,
                "added_queries": added.tolist(),
                "restoration_iterations": restoration_iterations,
                "restoration_failed": restoration_message,
            }
        )
    report = {
        "pad": [],
        "collision_queries": solves,
        "secondary": {"skipped": "pad_center_queries_are_fixed"},
    }
    return final_geometry, final, report


