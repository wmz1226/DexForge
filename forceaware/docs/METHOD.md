# ForceAware method

| Component | Implementation |
| --- | --- |
| Robot configuration and object pose tracking | `mpc_loss.py`: `_hand_reference_squared`, `_object_reference_squared`, `_reference_pose_loss` |
| Relative object rotation angle | `mpc_loss.py`: `_normalized_quaternion_angle` |
| Intermediate versus terminal tracking | `MpcObjective._allocate_training`, `_terminal_workspace`; the last intermediate tracking term is omitted |
| Material-point anchor alignment | `_query_world`, `_query_target_world`, `_marked_query_loss` |
| Distance-dependent anchor weight | `_prepare_auxiliary`; recomputed in forward, held fixed during backward |
| Contact-depth encouragement | `_physical_contact_loss` |
| Two penetration thresholds | `_penetration_loss` |
| Inter-finger capsule penalty | `_finger_joint_geometry`, `_finger_joint_collision_loss` |
| First-window bounded hand correction | `optimizer.py`: `_apply_initial_delta`; `workflow.py`: `_run_steps` |
| Initial correction penalty | `_initial_pose_regularization_loss` |
| First/second control differences | `_control_smoothness_loss` |
| Simulated hand velocity/acceleration penalty | `_state_smoothness_loss` |
| Position-target knot interpolation | `optimizer.py`: `_build_control_kernel`, `_build_dense_control_kernel` |
| Equivalent wrist-angle continuity | `wrist_reference.py`: `continuous_wrist_reference` |
| Reference initial pose and generalized velocity | `targets.py`: `initial_state` |
| Warm and velocity-compensated candidate initialization | `optimizer_contract.py`: `multistart_raws`; `servo_prior.py`: `ServoPriorOptimizer` |
| Differentiable rollout and parallel Adam | `WindowOptimizer._backward`, `_iteration` |
| Fixed-order state-adjoint accumulation | `native_adjoint/dynamics_vjp.py`, `native_adjoint/geometry_vjp.py` in the bundled simulator |
| Best finite iterate and candidate selection | `_track_best`, `_restore_best`, `_select_candidate` |
| Executed prefix and shifted warm start | `workflow.py`: `_run_steps`, `_shift_warm` |

The optimized variables are position-control targets, not independent contact
forces. The simulator computes actuation and contact forces. There is no
absolute control-effort penalty or reference-velocity tracking term. Collision
terms are soft penalties, and numerical validity is not a collision-feasibility
certificate. Only the first window optimizes the initial hand configuration;
subsequent windows use the actual preceding simulated state.

The first state uses the reference pose and velocity estimated from the next
source frame with quaternion-aware differencing. A held or final reference
starts at rest. Acceleration is determined by the simulator. Equivalent wrist
branches are made continuous within joint limits before the existing linear
interpolation in actuator coordinates; source-frame physical poses are preserved.

Half of the candidates retain warm-start initialization, including one
unperturbed candidate. The remainder initialize the control targets at
`q_ref + (kd / kp) * dq_ref`, using the scene's position-servo parameters and
reference-knot velocities. The original perturbations and candidate selection
remain in effect. This compensates servo damping without an inverse-dynamics
solve or a change to the objective.

The simulator defines the contact-frame derivative policy in `collision_config.py`.
For GS + hard (`contact_topk = 1`), its query API zeroes normal Jacobian rows and its
native collision adjoint drops the frame cotangent (normal and both tangents).
Distance and contact-position derivatives remain, including the normal dependence
inside the contact-position formula. GS + soft and both mesh modes retain physical
normal and full contact-frame derivatives, including tangent-frame spin. Selection is per collision batch, including mixed scenes.

The hard-GS gradient mask leaves its forward geometry unchanged. This policy applies to native
simulation, terminal observations and the geometry API consumed by ContactAware.
ForceAware configures this policy through the simulator. Hard-GS frame stopping
is a surrogate gradient; forward geometry and distance/contact-point derivatives
are preserved.

Simulator physical parameters are
fixed during MPC. Parameter identification is not part of this controller.
Shared state-gradient contributions are gathered in a fixed order without
changing the forward simulation or local derivative formulas. This improves
repeatability on the same device and runtime; contact switching remains nonsmooth.

Component normalization and conversion to millimeters are explicit in
`mpc_loss.py`, `control.py`, and the YAML profiles. Those factors are absorbed in
the matrices and coefficients used in the paper equations.

# Collision targets and contact fusion

The simulator (`third_party/comfree_warp`) has one hand-object collision path, used by the
differentiable step, the generic `comfree_warp.step` (object settling) and ContactAware's point
queries. Callers set `contact_physics.contact_topk = k`; the target kind comes from
the scene XML. For GS soft, k selects initial seeds, while the field uses complete
positive support with a fixed 1 mm width.

| Scene object geom | Target components |
| --- | --- |
| `type="gs"` (Gaussian PLY or sphere NPZ) | the Gaussian spheres |
| `type="sdfmesh"` (closed OBJ, welded on load) | one per point: its nearest triangle, signed by the angle-weighted pseudo-normal of the closest feature (exact for closed meshes) |

Plane supports use the target's static spheres: the Gaussians, or the mesh's convex-hull vertices.

**Features** (`collision_targets.py`). A candidate stores the component's feature in the target
body frame, `anchor = (a, r)` and `axis = (e, code)`, `code = side * kind`:

| kind | feature | closest point `c` | distance of a source sphere (centre `s`, radius `r_s`) |
| --- | --- | --- | --- |
| point | Gaussian sphere, or mesh vertex (`r = 0`) | `a` | `side * |c - s| - r - r_s` |
| edge | mesh edge line, unit direction `e` | `a + e e.(s - a)` | `side * |c - s| - r_s` |
| face | mesh face plane, outward normal `e` | `s - e e.(s - a)` | `e.(s - a) - r_s` |

`side` is -1 when the point is inside the mesh (always +1 for spheres). The inward normal is
`side * unit(c - s)` (`-e` for a face) and the contact position is `c - n r`. The broadphase fixes
the feature; within it these are the exact signed distance, closest point and normal of the mesh,
and their derivatives with respect to both body poses are exact (the target body receives
`outer(g_anchor, a) + outer(g_axis, e)` on its rotation). There is no approximation parameter.

**Hard and mesh selection** (`native_adjoint/gaussian_collision.py`). Each hand collision group owns one slot. GS soft reuses this search for its initial seeds.

1. `_slot_bounds`: the slot's previous `k + 1` candidate pairs, re-evaluated at the current pose,
   are distinct pairs of the current configuration, so their largest distance bounds the slot's
   (k+1)-th smallest pair distance. It only tightens pruning (a missing candidate gives the
   slot threshold).
2. `_source_nearest`: every source finds its nearest pair within the bound; a source is skipped by
   the target's bounding box only when its gap to the box exceeds both zero and the reach (the
   reach is negative under penetration). Below-bound sources are compacted per slot.
3. `_select_slots` (one tile per slot): lanes rank the compacted sources, the `k + 1` sources
   nearest by their own nearest pair are expanded to their `k + 1` nearest components, and lists
   merge under the total order (distance, component, source), the tie order of the original
   Gaussian broadphase. A source in the top pairs is among those `k + 1` sources, so the result is
   exact and independent of scheduling.

Source positions in the target frame, sphere pair distances, the sphere contact and its VJP use the
original Gaussian expressions and operation order: with `k = 1` on a Gaussian scene, contacts and
raw distance/position derivatives retain the original formulas. The configured hard-GS
frame stop changes the optimization gradient while leaving forward contacts unchanged.

**Ranked mesh fusion** (`contact_fusion.py`). Rank `k` is the boundary `d_b` (the slot threshold if missing).
Weights `a_i = max(d_b - d_i, 0)` blend distance, contact position and inward normal, and the normal
is renormalized. A candidate enters or leaves the top k with zero weight, so the fused contact is
continuous; the fused distance is a weighted mean, never deeper than the candidates; weights are
scale-free (no temperature). When at most one candidate carries weight (always for `k = 1`) or the
blended normal degenerates, the nearest pair is returned exactly.

**Ranked fusion VJP** (`contact_fusion.fuse_vjp`, `geometry_vjp.py`). Blended outputs distribute their seeds
analytically to every weighted candidate and to the boundary (`dD/dd_i = (a_i - (d_i - D)) / A`,
`dD/dd_b = sum_i (d_i - D) / A`, likewise for position and the normal through its normalization);
the exact nearest pair passes the distance, position and contact-frame seeds straight through the
pair cost. Each pair cost (`sphere_cost` for spheres, `feature_cost` for mesh features) is
differentiated with `wp.grad`; differentiated code never reassigns a returned tuple inside a
branch, whose Warp adjoint is wrong on the CPU backend. Gradients reach bodies in slot-then-rank
order without atomics, so they repeat exactly.

**Continuous GS soft** (`smooth_contact.py`, `native_adjoint/continuous_contact.py`,
`native_adjoint/continuous_query.py`). The field is the minimum of
`sum(w_i*d_i) + 2h/3*(sum(w_i**1.5)-1)` with `w >= 0`, `sum(w) = 1`, `h = 1 mm`.
Its positive weights are `((tau-d_i)/h)^2`. A subset threshold bounds the complete
root from above; a conservative BVH range search gathers all positive support.
Warm queries re-evaluate cached pairs, with the native nearest-pair search as cold
fallback. Support overflow is refined, never silently truncated. The model carries
up to 512 pairs per slot and reports unresolved overflow or degenerate frames.

Analytic implicit differentiation propagates distance, position and full frame
seeds through this field. Float64 CA point queries reuse the same threshold,
value and derivative functions with the simulator's target/BVH. Complete support
eliminates the k+1 rank-boundary derivative of the earlier GS soft model. Physical
activation, force projection and other simulator switches can still be nonsmooth;
this is not a claim that every long rollout is globally smooth.

**Numerical details.** Quaternion integration uses the correct small-angle
limit at zero angular velocity. The positive-beta force path computes the original
softplus force at the final response and differentiates that expression. Default
optimization uses beta=0.
