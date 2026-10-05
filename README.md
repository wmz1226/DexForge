<h1 align="center">DexForge: High-Fidelity Physics-Informed Dexterous Retargeting</h1>

<p align="center">
  <a href="https://scholar.google.com/citations?hl=zh-CN&amp;user=oElJeMUAAAAJ">Meizhong Wang</a><sup>1</sup>,
  <a href="https://ruiqini.github.io/">Ruiqi Ni</a><sup>3</sup>,
  <a href="https://ntu-caokun.github.io/">Kun Cao</a><sup>1,2,*</sup>,
  <a href="https://personal.ntu.edu.sg/elhxie/">Lihua Xie</a><sup>4</sup>,
  <a href="https://scholar.google.com/citations?user=QUTN3IwAAAAJ&amp;hl=zh-CN">Yiguang Hong</a><sup>1,2</sup>
</p>

<p align="center">
  <sup>1</sup> Tongji University &nbsp;&nbsp;
  <sup>2</sup> Shanghai Research Institute for Intelligent Autonomous Systems<br>
  <sup>3</sup> Purdue University &nbsp;&nbsp;
  <sup>4</sup> Nanyang Technological University, Singapore
</p>

<p align="center">
  <sup>*</sup> Corresponding author: <a href="https://ntu-caokun.github.io/">Kun Cao</a>
  (<a href="mailto:caokun@tongji.edu.cn">caokun@tongji.edu.cn</a>)
</p>

<p align="center">
  <a href="https://wmz1226.github.io/DexForge/"><strong>Project Page</strong></a>
</p>

<p align="center">
  <a href="docs/assets/pipeline.svg">
    <img src="docs/assets/pipeline.svg" width="100%" alt="DexForge pipeline: hand-object reconstruction, ContactAware retargeting, and ForceAware dynamics optimization.">
  </a>
</p>

This repository contains the **retargeting component** of DexForge. It converts
prepared MANO hand-object demonstrations into robot hand trajectories and
actuator targets through two stages:

- **ContactAware (CA):** kinematic retargeting and contact refinement.
- **ForceAware (FA):** dynamics optimization using CA trajectories and guidance.

Both stages share the bundled ComFree-Warp simulator, which supports mesh and
Gaussian sphere (GS) geometry with hard or soft contact modes.

## Setup

Requires Linux, Python 3.10, an NVIDIA GPU, EGL, and a C++ compiler.

```bash
git clone https://github.com/wmz1226/DexForge.git
cd DexForge
bash setup.sh
source .venv/bin/activate
```

Download the required MANO model into `assets/mano/` following its
[setup instructions](assets/mano/README.md). Model files are not included.
An existing compatible Python environment can also be used. `setup.sh` accepts
`DEXFORGE_PYTHON` and `DEXFORGE_VENV` to select the interpreter and environment.

## Quickstart

Run ContactAware followed by ForceAware on the included Sugar Box sequence with
LEAP Hand:

```bash
python run_example.py --scene gs --mode hard
```

Use `--scene mesh` for mesh geometry and `--mode soft` for soft contact. These
options can be combined; all four configurations use the same settings across
both stages. The first run compiles GPU kernels.

Results are saved to `results/<scene>_<mode>/<sequence>/`:

| Result | Path within the output directory |
|---|---|
| ContactAware video | `contactaware/retarget.mp4` |
| ForceAware video | `forceaware/rollout.mp4` |
| ForceAware metrics | `forceaware/metrics.json` |
| Wall time for each stage | `timings.json` |

The runner copies the input sequence before processing it. For another run,
choose a new `--output-dir /path/to/result`. Use `--sequence-dir /path/to/sequence`
and `--hand leaphand` for your own [prepared inputs](contactaware/README.md#inputs),
or `--mano-models /path/to/models` for externally stored MANO models.

## Documentation

- [ContactAware](contactaware/README.md): input formats, kinematic retargeting, and contact guidance.
- [ForceAware](forceaware/README.md): dynamics optimization, configuration, outputs, and replay.
- [Simulator collision API](forceaware/third_party/comfree_warp/comfree_warp/geometry/README.md): mesh/GS queries, contact modes, and gradients.

## License

Original project code is released under the [MIT License](LICENSE). Bundled
third-party software and example data retain their own licenses, including the
ComFree Core Academic Research License; see [third-party notices](THIRD_PARTY_NOTICES.md).
