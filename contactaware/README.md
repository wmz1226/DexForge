# ContactAware

ContactAware is the kinematic stage of DexForge retargeting. It converts prepared
MANO demonstrations into robot trajectories and refined contact guidance for FA.
Use the shared environment described in the [root README](../README.md).

## Usage

From the repository root:

```bash
python contactaware/retarget.py --sequence-dir /path/to/sequence \
  --hand leaphand --output-dir /path/to/ca_result
```

The default output is `<sequence>/retarget/<hand>/contactaware`. MANO models are
loaded from `assets/mano/`, or `MANO_MODEL_ROOT`. Robot assets are loaded from
`assets/`, or `HAND_ASSETS_ROOT`.

All solver parameters are in [configs/contactaware.yaml](configs/contactaware.yaml).
Use a grouped YAML file with `--config` for overrides, or repeat `--set KEY=VALUE`:

```bash
python contactaware/retarget.py --sequence-dir /path/to/sequence \
  --hand leaphand --output-dir /path/to/ca_result \
  --set scene=gs --set contact_topk=8
```

`scene=gs` selects Gaussian geometry; `scene=sdf` selects the mesh collision scene.
`contact_topk=1` selects hard contact and values 2-8 select soft contact. CA queries
and its object-settling simulation use the same configuration. Use the root
`run_example.py --scene gs --mode soft` entry point to apply this configuration
to both CA and FA.

## Inputs

```text
sequence/
  mano_raw/meta.json
  mano_raw/hand_pose_51.npy
  mano_raw/hand_shape_10.npy
  mano_raw/object_pose_camera_7.npy
  mano_raw/camera_pose_7.npy
  mano_raw/extrinsics_4x4.npy
  scene/<hand>/<object>_gs.xml
  scene/<hand>/<object>_sdf.xml
  scene/<hand>/<object>_mesh.xml
```

Prepared DexYCB, HOT3D, and V2D sequences are supported. Metadata identifies the
source, object, camera, and hand side. Hand poses contain global orientation,
45 PCA coefficients, and translation. Object/camera poses use
`[qx, qy, qz, qw, tx, ty, tz]`; distances are in meters. Hand topologies are
registered in `contactaware/models/hand_specs.py`.

Standalone CA writes wrist-mounted scene XMLs into the input sequence. The root
runner copies the sequence first. FA must use the same mounted scene and joint
ordering as the saved CA trajectory.

## Outputs

| File | Content |
|---|---|
| `<hand>_qpos.npy` | Robot trajectory in world coordinates |
| `object_pose_7.npy`, `camera_pose_7.npy` | Object and camera poses |
| `contact_guidance.npz` | Contact masks, refined anchors, and keyframes for FA |
| `metrics.json`, `config.json`, `stage_timings.json` | Metrics, resolved settings, and timings |
| `retarget.mp4`, `keyframe/*.png` | Trajectory video and keyframe renders |

CA uses the simulator's [collision query API](../forceaware/third_party/comfree_warp/comfree_warp/geometry/README.md).
Soft constraints, anchor projection, and line searches use the configured fused
field and its distance derivative. Keyframe SQP and restoration use the
`sqp_stagnation_patience` and `sqp_progress_relative_tolerance` settings.
