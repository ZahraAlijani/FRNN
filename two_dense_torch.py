"""Dense 2D-NN-2L: the paper's own shallow two-layer dense model (Section
"2D Neural Networks", Tables 1-3 in TwoD_NN.tex), not the patch-recurrent 2D-RNN.

    Layer 1: Y = ReLU(W_L1 X W_R1 + B1),   X in R^{28x28}, Y in R^{mu2 x mu2}
    Layer 2: a = softmax(W_L2 Y W_R2 + B2), a in R^{10x1}

Param count: 2(28*mu2) + mu2^2 + 10*mu2 + mu2 + 10 = mu2^2 + 67*mu2 + 10
(matches TwoD_NN.tex line 282-286 exactly). Same custom margin loss as the
2D-RNN (Eq. CusCatLoss, gamma=1, m+=0.9, m-=0.1). Xavier init, initial
learning rate eta = mean_initial_loss / 3 (per TwoD_NN.tex Section
"Experimental Results", held fixed thereafter -- the paper does not specify
a schedule beyond the initial value).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DenseNet2L:
    W_L1: torch.Tensor
    W_R1: torch.Tensor
    B1: torch.Tensor
    W_L2: torch.Tensor
    W_R2: torch.Tensor
    B2: torch.Tensor

    def parameters(self) -> list[torch.Tensor]:
        return [self.W_L1, self.W_R1, self.B1, self.W_L2, self.W_R2, self.B2]


def _xavier(shape, gen, device, dtype):
    fan_in, fan_out = shape[1], shape[0]
    bound = (6.0 / (fan_in + fan_out)) ** 0.5
    t = (torch.rand(*shape, generator=gen, dtype=dtype) * 2 - 1) * bound
    return t.to(device).requires_grad_(True)


def create_dense_net(mu2: int, in_dim: int = 28, seed: int = 0,
                      device: str = "cuda", dtype: torch.dtype = torch.float64) -> DenseNet2L:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return DenseNet2L(
        W_L1=_xavier((mu2, in_dim), gen, device, dtype),
        W_R1=_xavier((in_dim, mu2), gen, device, dtype),
        B1=torch.zeros(mu2, mu2, dtype=dtype, device=device, requires_grad=True),
        W_L2=_xavier((10, mu2), gen, device, dtype),
        W_R2=_xavier((mu2, 1), gen, device, dtype),
        B2=torch.zeros(10, 1, dtype=dtype, device=device, requires_grad=True),
    )


def forward_batch(net: DenseNet2L, images: torch.Tensor) -> torch.Tensor:
    """images: (N, 28, 28) -> prob (N, 10, 1)."""
    y1 = torch.relu(torch.einsum("mp,npq,qc->nmc", net.W_L1, images, net.W_R1) + net.B1)
    logit = torch.einsum("cm,nmk,ko->nco", net.W_L2, y1, net.W_R2) + net.B2
    return torch.softmax(logit, dim=1)


def margin_loss_batch(prob: torch.Tensor, onehot: torch.Tensor,
                       gamma: float = 1.0, m_plus: float = 0.9, m_minus: float = 0.1) -> torch.Tensor:
    per_sample = (onehot * torch.clamp(m_plus - prob, min=0.0) ** 2
                  + gamma * (1.0 - onehot) * torch.clamp(prob - m_minus, min=0.0) ** 2).sum(dim=1)
    return per_sample.mean()


def param_count(mu2: int) -> int:
    return mu2 * mu2 + 67 * mu2 + 10
