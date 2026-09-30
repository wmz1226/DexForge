# Collision geometry API

The simulator owns scene loading, target selection, fusion, derivatives and surface
projection. ContactAware consumes this API; ForceAware configures the same collision
model before compiling native simulation and observation adjoints.

```python
from comfree_warp import CollisionConfig, configure_collision, load_model
from comfree_warp.geometry import ObjectCollisionQuery, ObjectGeometry

config = CollisionConfig(contact_topk=8, distance_offset=0.0)
query = ObjectCollisionQuery(scene_xml, config=config)
phi_and_gradient = query.distance_features(points_in_object_frame)  # N x 4
geometry = ObjectGeometry(query)
values, rows = geometry.compact(points_in_object_frame)
selected_jacobians = rows[selected_points]  # Nselected x 7 x 3
cloud_results = query.query_clouds([mano_finger_points, other_points])
# Each result is (per-point values, lazy per-point Jacobians), with zero radii.
# Optional source_radii accepts a scalar or one radius array for each cloud.

model, cpu_model = load_model(str(scene_xml))
configure_collision(model, config)  # before native runtime compilation
```

The scene selects `gs` or `sdfmesh`; `contact_topk=1` selects hard, 2–8 selects
fused soft. Values have columns `[distance, surface_x/y/z, outward_normal_x/y/z]`.
`distance_features` returns the actual distance derivative, not a unit normal.
The legacy callable query returns `(distance, surface, inward_normal)` for the
existing contact adapter. Every caller uses the compiled target's float32 BVH
through `collision_targets.nearest_components` and conservative range traversal.
Hard and mesh use `contact_fusion`; complete-support GS soft uses `smooth_contact`,
shared between float64 solver queries and float32 native simulation. Previous
candidates only provide a conservative search bound, never a substitute for
complete current support.

Runtime sources are centers with optional nonnegative `source_radii`; omitted
radii mean zero, so a point is the zero-radius Gaussian case. Both target types
accept these runtime sources without defining each point in XML. The XML supplies
the target and its BVH. Queries evaluate the configured field outside the simulator's active range as well,
so distant points still receive useful geometric values and gradients.

For mesh targets, native selection supplies one nearest triangle feature per
source. Soft fusion can blend multiple source contacts in a simulation slot;
raising k for a single mesh query point does not invent extra triangle candidates.

| Target / mode | Distance derivative | Surface-point derivative | Normal derivative |
|---|---|---|---|
| GS / hard | retained | retained | stopped |
| GS / soft | retained | retained | retained |
| mesh / hard | retained | retained | retained |
| mesh / soft | retained | retained | retained |

In GS hard mode, the surface normal can jump when the active Gaussian changes,
making its gradient unreliable over finite optimization steps. Only normal/frame
output derivatives are stopped in this mode; distance and surface-point
derivatives remain active. The normal's role inside the surface-point formula
remains differentiable. The native collision layer applies the same policy per
batch to dynamics, observations, and its Warp tape. Forward values do not change.
Raw internal primitives retain exact derivatives for testing.

Soft constraints reference the fused field and reselect candidates on evaluation.
Soft surface projection solves its zero set with that field's true gradient. The
bound returned by `distance_bounds` only schedules swept queries; it never rejects
a state. For GS soft the configured distance is itself a valid lower bound; using
the hard minimum as a lower bound would be incorrect for this smooth minimum.
Ground supports retain the simulator's
shared target/plane rule in both modes.

GS soft minimizes `<w,d> + 2h/3 (sum(w**1.5)-1)` on the probability simplex,
with `h = 0.001 m`. Positive weights are `((tau-d)/h)^2`, where their sum is one.
`contact_topk` controls initial BVH seeds. All positive support is retained up to
512 pairs per slot/query, with conservative overflow refinement; unresolved
overflow or a degenerate normal raises an error. The normal is the normalized
weighted pair normal; the distance derivative is the envelope derivative.

Hard's reusable buffers, warm starts and BVH are shared. GS soft adds range
traversal when seeds do not cover its support. Lazy surface/normal rows are formed
only when requested. Neither CA nor FA implements geometry-specific selection or
fusion formulas.
