# Input format

Use a prepared robot reference and matching collision/visual scenes. The controller
does not estimate hand poses or initialize contact correspondences from images.
All arrays must be finite numeric arrays loadable with `allow_pickle=False`.

## Reference arrays

Let `T >= 2` be the reference frame count, `D` the number of scalar hand/base
joints, and `C >= 1` the number of contact slots.

| File / field | Shape | Meaning |
| --- | --- | --- |
| `<hand>_qpos.npy` | `(T, D)` | Robot configuration in scene joint order; translations in meters, angles in radians |
| `object_pose_7.npy` | `(T, 7)` | Object-to-world pose `[qx, qy, qz, qw, tx, ty, tz]`; translation in meters |
| `camera_pose_7.npy` | `(T, 7)` | Camera-to-world pose in the same quaternion/translation order; first camera pose is used for rendering |
| `contact_guidance.npz:contact_mask` | `(T, C)` | Active contact indicators |
| `contact_guidance.npz:contact_weight` | `(T, C)` | Nonnegative guidance weights |
| `contact_guidance.npz:contact_pos_obj` | `(T, C, 3)` | Refined anchors in the object's local frame, meters |
| `contact_guidance.npz:hand_query_body_ids` | `(T, C)` | Integer MuJoCo body IDs of the robot material points |
| `contact_guidance.npz:hand_query_local_pos` | `(T, C, 3)` | Material-point positions in those body frames, meters |

Contact body IDs must refer to the exact scene model. They are not joint IDs or
fingertip indices. Rebuilding a scene with different body ordering requires
remapping these IDs. The code maps the active query bodies to simulator Gaussian
collision outputs and rejects unsupported mappings.

The contact mask selects guidance targets; it does not disable other physical
hand-object contacts.

`mano_raw/meta.json` needs only the object identifier and camera field of view:

```json
{"object": "example", "camera": {"fovy": 45.0}}
```

For `object = example`, the collision scene is `scene/<hand>/example_gs.xml`
or `example_sdf.xml`, and the visual scene is `example_mesh.xml`. A numeric dataset prefix followed by an underscore is
removed when deriving the scene stem. The `mano_raw` directory supplies metadata and camera poses; FA does not load
MANO models.

The camera pose uses optical camera axes (x right, y down, z forward). Rendering
converts them to the MuJoCo camera convention. The field of view is vertical, in
degrees. `FORCEAWARE_CAMERA_FOVY` is an optional rendering-only override.

## Scene contract

The hand must have scalar joints ordered before the object's free joint, named
`obj_joint`. The first six hand coordinates are base translation (three) and
base rotation (three), followed by finger joints. Each hand joint requires a
corresponding position actuator. The object is unactuated. The conventional
visual scene must preserve the same body and joint ordering as the Gaussian
scene. XML asset references should be relative and must remain valid when the
sequence is relocated.

The three base-rotation joints must be orthogonal, co-located hinges on one
body or an identity-transform serial chain. Their equivalent angle branches
must admit a continuous representation within the joint limits.

Supported hand topology descriptors are in `forceaware/hand_specs.py`. They
specify a palm body and fingertip sites, from which the code derives finger body
chains. Every physical finger must include Gaussian collision-bearing bodies.
Capsule fitting requires at least three source collision spheres per fitted
body. A new embodiment can provide its corresponding topology descriptor.

The bundled loader accepts `geom type="gs" file="..."` sphere clouds. NPZ
clouds store `local_pos` with shape `(N,3)` and `radius` with shape `(N,)`, with
positive radii; clouds may also include `local_normal`. Supported
Gaussian PLY input is handled by `gaussian_loader.py`. Collision masks, friction,
mass, inertia, damping and actuator parameters come from the scene. They are not
estimated during MPC.

## Time conventions

`ref_dt` specifies reference frame spacing; `mpc_dt` is the planner physics step;
`exec_dt` is the executor step; `knot_dt` is control-knot spacing; and `action_dt`
is the executed prefix/replanning interval. The validated integer ratios define
the dense rollout. Configuration `optimizer.horizon` counts control knots.
Object quaternion interpolation is normalized and sign-aligned. References are
held at their final frame when the prediction extends beyond the sequence.

For an offline moving clip, the initial hand and object velocities are estimated
from the start pose and next source-frame pose using `ref_dt`. Hold mode and a
start at the final frame use zero velocity. Later windows use the simulated state.

The recorded `qpos_traj` follows MuJoCo conventions: the object's free-joint
state is `[tx, ty, tz, qw, qx, qy, qz]`. This differs from the input pose-array
layout. `ctrl_traj` is the sequence of dense actuator position targets.
