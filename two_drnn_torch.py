"""PyTorch batched implementation of the 2D-RNN, mirroring two_drnn.py exactly.

Forward equations are transcribed 1:1 from two_drnn.py / forward_2DRNN.m (verified
bit-exact against a captured MATLAB reference in matlab.mat). Gradients are obtained
via autograd instead of a hand-derived backward pass, since the batched manual
backward (two_drnn_fast.py) is a second, independent implementation that would need
its own verification -- autograd removes that risk entirely by construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TorchNet:
    W_FeatL: torch.Tensor
    W_FeatR: torch.Tensor
    B_Feat: torch.Tensor
    W_HP_L: torch.Tensor
    W_HP_R: torch.Tensor
    W_H_L: torch.Tensor
    W_H_R: torch.Tensor
    B_H: torch.Tensor
    W_InfL: torch.Tensor
    W_InfR: torch.Tensor
    B_Inf: torch.Tensor
    p: int
    f: int
    h1: int
    h2: int
    m_grid: int

    def parameters(self) -> list[torch.Tensor]:
        return [self.W_FeatL, self.W_FeatR, self.B_Feat, self.W_HP_L, self.W_HP_R,
                self.W_H_L, self.W_H_R, self.B_H, self.W_InfL, self.W_InfR, self.B_Inf]


def create_net_torch(p: int, f: int, h1: int, h2: int, m: int, seed: int,
                      device: str, dtype: torch.dtype = torch.float64) -> TorchNet:
    """Same scheme as create_net_2DRNN.m: scale_init=0.01, N(0,1) weights, zero biases."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scale_init = 0.01

    def randn(*shape):
        return (scale_init * torch.randn(*shape, generator=gen, dtype=dtype)).to(device).requires_grad_(True)

    def zeros(*shape):
        return torch.zeros(*shape, dtype=dtype, device=device, requires_grad=True)

    return TorchNet(
        W_FeatL=randn(f, p), W_FeatR=randn(p, f), B_Feat=zeros(f, f),
        W_HP_L=randn(h1, f), W_HP_R=randn(f, h2),
        W_H_L=randn(h1, h1), W_H_R=randn(h2, h2), B_H=zeros(h1, h2),
        W_InfL=randn(10, h1), W_InfR=randn(h2, 1), B_Inf=zeros(10, 1),
        p=p, f=f, h1=h1, h2=h2, m_grid=m,
    )


def forward_batch(net: TorchNet, images: torch.Tensor):
    """images: (N, 28, 28). Returns prob (N,10,1), h_final (N,h1,h2)."""
    p, m = net.p, net.m_grid

    patch_store = [[None] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            x = images[:, i * p:(i + 1) * p, j * p:(j + 1) * p]
            z = torch.einsum("fp,npq,qg->nfg", net.W_FeatL, x, net.W_FeatR) + net.B_Feat
            patch_store[i][j] = torch.relu(z)

    h_prev = torch.zeros(images.shape[0], net.h1, net.h2, dtype=images.dtype, device=images.device)
    for i in range(m):
        for j in range(m):
            pc = patch_store[i][j]
            pre = (torch.einsum("af,nfg,gb->nab", net.W_HP_L, pc, net.W_HP_R)
                   + torch.einsum("ac,ncd,db->nab", net.W_H_L, h_prev, net.W_H_R)
                   + net.B_H)
            h_prev = torch.sigmoid(pre)

    h_final = h_prev
    logit = torch.einsum("ah,nhb,bc->nac", net.W_InfL, h_final, net.W_InfR) + net.B_Inf
    prob = torch.softmax(logit, dim=1)
    return prob, h_final


def margin_loss_batch(prob: torch.Tensor, onehot: torch.Tensor,
                       gamma: float = 1.0, m_plus: float = 0.9, m_minus: float = 0.1) -> torch.Tensor:
    """Sum over classes, mean over batch -- standard DL per-sample-averaged loss."""
    per_sample = (onehot * torch.clamp(m_plus - prob, min=0.0) ** 2
                  + gamma * (1.0 - onehot) * torch.clamp(prob - m_minus, min=0.0) ** 2).sum(dim=1)
    return per_sample.mean()
