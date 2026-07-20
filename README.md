# 2D-NN

Python/PyTorch port of an ultra-lightweight 2D-RNN for MNIST digit recognition, plus a
GPU-batched training pipeline and an `f`-parameter sweep.

## Contents

- `two_drnn.py` — per-sample NumPy forward/backward/Adam, a direct transcription of the
  original MATLAB (`*_2DRNN.m`), verified bit-exact against a captured MATLAB reference.
- `two_drnn_torch.py`, `train_torch.py` — PyTorch/GPU batched training (standard minibatch
  Adam), forward pass and autograd gradients verified against the same MATLAB reference.
- `run_sweep.py`, `build_table.py` — sweep driver over `f` and LaTeX/CSV table generation.
- `run_custom.py` — single custom-hyperparameter training run.
- `*.m` — original MATLAB source.

## Credits

The model architecture and original MATLAB implementation (`create_net_2DRNN.m`,
`forward_2DRNN.m`, `backward_2DRNN.m`, `adam_update_2DRNN.m`, `TrainExample.m`,
`Example2.m`) are by **Babak Shiri** (Neijiang Normal University, ORCID
[0000-0003-2249-282X](https://orcid.org/0000-0003-2249-282X)):

> Shiri, Babak. *MATLAB code and model weights for ultra-lightweight 2D neural
> networks.* Zenodo, 2026. https://doi.org/10.5281/zenodo.20169150

Supporting the paper *"2D Neural Networks: Application on Ultra-Lightweight Image
Recognition on Mobile Devices."*

See [CONTRIBUTORS.md](CONTRIBUTORS.md).
