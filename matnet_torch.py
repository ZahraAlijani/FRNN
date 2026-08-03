"""MatNet baseline, following the primary source directly (PDF read page-by-page,
not a secondary summary): Gao, J.; Guo, Y.; Wang, Z. "Matrix Neural Networks."
arXiv:1601.03805 / Advances in Neural Networks - ISNN 2017, LNCS vol. 10261,
pp. 313-320.

Hidden layer (Eq. 2.2):      X^(l+1) = sigmoid(U^(l) X^(l) (V^(l))^T + B^(l))
Output layer (Eq. 2.6-2.7, "Classification network II" -- confirmed used for
the MNIST experiment in Sec. 4.1): flatten the last hidden matrix with vec(),
then a standard linear + softmax head:
    n_k = vec(X^(L))^T u_k + tb_k,   o = softmax(n)
Loss (Eq. 2.9): standard cross-entropy (not our margin loss).

Confirmed from the paper's Sec. 4.1 (quoted numbers, not paraphrased):
  - Baseline: 2 hidden layers, 20x20 and 16x16, sigmoid, sparsity/weight-decay
    OFF -> 5,536 total parameters, 97.3% MNIST test accuracy.
  - Best variant found in their sweep: depth 4 (two hidden layers), 160x160
    each -> 316,160 parameters, 98.48% test accuracy.
Our from-equations reimplementation's own parameter count may differ slightly
from their reported 5,536 (some detail of their exact layer wiring isn't
recoverable from the paper text alone) -- we report both their published
numbers and our own reproduction's numbers separately, not as one merged row.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MatNet:
    layers: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]  # (U, V, B) per hidden layer
    out_W: torch.Tensor   # (10, h_last*h_last)
    out_b: torch.Tensor   # (10,)
    last_h: int

    def parameters(self) -> list[torch.Tensor]:
        ps: list[torch.Tensor] = []
        for U, V, B in self.layers:
            ps += [U, V, B]
        ps += [self.out_W, self.out_b]
        return ps


def _xavier(shape, gen, device, dtype):
    fan_in, fan_out = shape[1], shape[0]
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    t = (torch.rand(*shape, generator=gen, dtype=dtype) * 2 - 1) * bound
    return t.to(device).requires_grad_(True)


def create_matnet(hidden_dims: list[int], in_dim: int = 28, seed: int = 0,
                   device: str = "cuda", dtype: torch.dtype = torch.float64) -> MatNet:
    """hidden_dims e.g. [20, 16] -> two hidden bilinear layers of size 20x20, 16x16."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    layers = []
    prev = in_dim
    for h in hidden_dims:
        U = _xavier((h, prev), gen, device, dtype)
        V = _xavier((h, prev), gen, device, dtype)
        B = torch.zeros(h, h, dtype=dtype, device=device, requires_grad=True)
        layers.append((U, V, B))
        prev = h
    out_W = _xavier((10, prev * prev), gen, device, dtype)
    out_b = torch.zeros(10, dtype=dtype, device=device, requires_grad=True)
    return MatNet(layers=layers, out_W=out_W, out_b=out_b, last_h=prev)


def forward_batch(net: MatNet, images: torch.Tensor) -> torch.Tensor:
    """images: (N, 28, 28) -> prob (N, 10)."""
    x = images
    for U, V, B in net.layers:
        x = torch.sigmoid(torch.einsum("hp,npq,gq->nhg", U, x, V) + B)
    flat = x.reshape(x.shape[0], -1)                       # vec(X^(L)), (N, h*h)
    logit = flat @ net.out_W.T + net.out_b                 # (N, 10)
    return torch.softmax(logit, dim=1)


def param_count(hidden_dims: list[int], in_dim: int = 28) -> int:
    n = 0
    prev = in_dim
    for h in hidden_dims:
        n += 2 * h * prev + h * h
        prev = h
    n += 10 * prev * prev + 10
    return n
