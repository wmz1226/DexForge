# manopth runtime subset

This copy contains the MANO loading and PyTorch hand-model code used by DexForge,
from [hassony2/manopth](https://github.com/hassony2/manopth), commit
`4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b`.

DexForge installs dependencies through its [root setup](../../../README.md) and
loads MANO model files from `assets/mano/` or `MANO_MODEL_ROOT`. Models retain the
[MANO license](https://mano.is.tue.mpg.de/license.html). The bundled [LICENSE](LICENSE)
and copyright notices in `mano/webuser/` apply to their respective code.

Original work: Javier Romero, Dimitrios Tzionas and Michael J. Black,
*Embodied Hands: Modeling and Capturing Hands and Bodies Together*, SIGGRAPH Asia
2017. The PyTorch port accompanies Yana Hasson et al., *Learning Joint Reconstruction
of Hands and Manipulated Objects*, CVPR 2019. Rotation utilities retain their
upstream attribution to Zhang Xiong's PyTorch SMPL implementation.
