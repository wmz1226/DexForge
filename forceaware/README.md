# ForceAware

ForceAware is the dynamics stage of DexForge retargeting. It optimizes actuator
position targets from CA trajectories and contact guidance using the shared
simulator and multi-start MPC. Use the environment in the [root README](../README.md).

## Usage

From the repository root, using the sequence processed by CA:

```bash
python forceaware/retarget.py --sequence /path/to/sequence --hand leaphand \
  --contactaware /path/to/ca_result --output /path/to/fa_result
```

`--output` must be a new directory. Without `--contactaware`, guidance defaults
to `<sequence>/retarget/<hand>/contactaware`. Scene assets, body IDs, and joint
ordering must match the CA output. See [input formats](docs/INPUTS.md).

## Configuration

[configs/forceaware.yaml](configs/forceaware.yaml) contains the default parameters.
Pass `--config /path/to/config.yaml` to use another configuration.

- `sequence.scene`: `gs` for Gaussian geometry or `sdf` for mesh collisions.
- `contact_physics.contact_topk`: 1 for hard contact, 2-8 for soft contact.
- `--contact-topk 8`: override the contact mode for both planning and execution.
- `FORCEAWARE_DEVICE=cuda:0`: select the simulation device.

The default planner uses 5 control knots, `knot_dt=0.06`, `mpc_dt=0.01`, and
`action_dt=0.03`: a 30-step prediction horizon, with replanning every 3 steps.
Each window uses 64 starts and 40 Adam updates. The simulator owns collision
selection, soft fusion, and gradient policy. Only GS hard stops normal/frame
output gradients; distance and contact-point gradients remain active.
See the [collision API](third_party/comfree_warp/comfree_warp/geometry/README.md)
for the gradient policy and soft fusion model.

## Outputs

`rollout.npz` records states, controls, candidate scores, validity, and timings.
`metrics.json` contains tracking and contact metrics; `config.json` contains the
resolved configuration; `rollout.mp4` shows the trajectory from the input camera.
See [method details](docs/METHOD.md) for objectives and contact handling.

Saved controls are also replayed in the independent MuJoCo mesh scene, writing
`replay/mujoco_cpu/replay.mp4`, `replay.npz`, and `replay.json`. This replay uses
its scene's own physical parameters. It does not determine acceptance of FA
optimization steps. In the bundled example, GS and mesh replay use different
object masses and friction coefficients.

To replay an existing result:

```bash
python forceaware/replay.py --rollout /path/to/fa_result/rollout.npz \
  --xml /path/to/sequence/scene/leaphand/sugar_box_mesh.xml \
  --contactaware /path/to/ca_result
```
