# Third-party materials

- **DexYCB:** The example trajectory is adapted from *DexYCB: A Benchmark for Capturing Hand Grasping of Objects*, Yu-Wei Chao et al., CVPR 2021. [Source](https://dex-ycb.github.io/), [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The supplied sample contains reconstructed MANO and object poses, camera calibration, and derived simulation inputs; machine-specific metadata has been removed.
- **YCB:** The Sugar Box mesh and texture originate from the Yale-CMU-Berkeley Object and Model Set; the Gaussian model is derived from this mesh. [Source and license](https://registry.opendata.aws/ycb-benchmarks/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **LEAP Hand:** The ten robot meshes match [MuJoCo Menagerie's LEAP Hand assets](https://github.com/google-deepmind/mujoco_menagerie/tree/main/leap_hand). Its MIT license is retained in [assets/leaphand/LICENSE](assets/leaphand/LICENSE). The scene descriptions are adapted for this simulator, with contact query points added.
- **MANO and manopth:** Obtain MANO model files from the [official website](https://mano.is.tue.mpg.de/download.php) under the [MANO license](https://mano.is.tue.mpg.de/license.html). The bundled [manopth](https://github.com/hassony2/manopth) runtime comes from commit `4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b` and retains its [GPLv3 license](contactaware/third_party/manopth/LICENSE). MANO loader files retain their original notices. These components are not relicensed under MIT.
- **Simulator:** See [forceaware/THIRD_PARTY_NOTICES.md](forceaware/THIRD_PARTY_NOTICES.md).

Third-party attributions identify upstream rights holders. The project MIT license does not relicense these materials.
