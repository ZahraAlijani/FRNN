"""Vectorized (batched) implementation of the 2D-RNN forward/backward passes.

This reproduces the exact math of ``two_drnn.py`` but processes a whole batch of
``N`` images at once with numpy tensor contractions instead of a per-sample
Python loop. The accumulated gradient over a batch is numerically identical
(up to floating point) to summing the per-sample gradients of the original code
-- this is checked in ``verify_fast.py``.

Update granularity is controlled by the caller: doing one Adam step per batch of
size ``B`` is minibatch training. ``B = 1`` recovers the original online update.
"""
from __future__ import annotations

import numpy as np

from two_drnn import TwoDRNN, _zero_buffers_like


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def forward_batch(net: TwoDRNN, images: np.ndarray):
    """Batched forward pass.

    ``images`` has shape (N, 28, 28). Returns prob (N,10,1), h_final (N,h1,h2),
    patch_store[i][j] (N,f,f) and hidden_history[i][j] (N,h1,h2).
    """
    p, m = net.p, net.m_grid
    N = images.shape[0]

    patch_store = [[None] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            x = images[:, i * p:(i + 1) * p, j * p:(j + 1) * p]        # (N,p,p)
            z = np.einsum("fp,npq,qg->nfg", net.W_FeatL, x, net.W_FeatR) + net.B_Feat
            patch_store[i][j] = _relu(z)                                # (N,f,f)

    hidden_history = [[None] * m for _ in range(m)]
    h_prev = np.zeros((N, net.h1, net.h2))
    for i in range(m):
        for j in range(m):
            pc = patch_store[i][j]
            pre = (np.einsum("af,nfg,gb->nab", net.W_HP_L, pc, net.W_HP_R)
                   + np.einsum("ac,ncd,db->nab", net.W_H_L, h_prev, net.W_H_R)
                   + net.B_H)
            h_prev = _sigmoid(pre)                                      # (N,h1,h2)
            hidden_history[i][j] = h_prev

    h_final = h_prev
    logit = np.einsum("ah,nhb,bc->nac", net.W_InfL, h_final, net.W_InfR) + net.B_Inf
    logit = logit - logit.max(axis=1, keepdims=True)
    exp = np.exp(logit)
    prob = exp / exp.sum(axis=1, keepdims=True)                        # (N,10,1)
    return prob, h_final, patch_store, hidden_history


def backward_batch(net: TwoDRNN, images: np.ndarray, onehot: np.ndarray,
                   prob: np.ndarray, h_final: np.ndarray,
                   patch_store, hidden_history) -> dict[str, np.ndarray]:
    """Batched backward pass. ``onehot`` has shape (N,10,1).

    Returns gradients summed over the batch (same shapes as the weights).
    """
    p, h1, h2, m = net.p, net.h1, net.h2, net.m_grid
    N = images.shape[0]
    gamma = net.loss["gamma"]; mp = net.loss["m_plus"]; mm = net.loss["m_minus"]

    grad = _zero_buffers_like(net)

    # Index labels: C=classes(10) A=h1 B=h2 F=f P=p O=1(trailing singleton), n=batch
    dL_dd = (-2.0 * onehot * np.maximum(mp - prob, 0.0)
             + 2.0 * gamma * (1.0 - onehot) * np.maximum(prob - mm, 0.0))   # (N,C,O)
    a = dL_dd * prob
    S = a.sum(axis=1, keepdims=True)
    logit_grad = a - prob * S                                              # (N,C,O)

    delta_inf = logit_grad
    # dW_InfL = delta_inf @ W_InfR.T @ h_final.T   -> (C,A)
    grad["dW_InfL"] = np.einsum("nco,bo,nab->ca", delta_inf, net.W_InfR, h_final)
    # dW_InfR = h_final.T @ W_InfL.T @ delta_inf   -> (B,O)
    grad["dW_InfR"] = np.einsum("nab,ca,nco->bo", h_final, net.W_InfL, delta_inf)
    grad["dB_Inf"] = delta_inf.sum(axis=0)

    # delta_h = W_InfL.T @ delta_inf @ W_InfR.T    -> (N,A,B)
    delta_h = np.einsum("ca,nco,bo->nab", net.W_InfL, delta_inf, net.W_InfR)

    for i in range(m - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            h_ij = hidden_history[i][j]
            p_ij = patch_store[i][j]
            x = images[:, i * p:(i + 1) * p, j * p:(j + 1) * p]

            if i == 0 and j == 0:
                h_prev = np.zeros((N, h1, h2))
            elif j == 0:
                h_prev = hidden_history[i - 1][m - 1]
            else:
                h_prev = hidden_history[i][j - 1]

            delta_pre_h = delta_h * (h_ij * (1.0 - h_ij))                  # (N,A,B)

            # dW_HP_L = delta_pre_h @ W_HP_R.T @ p_ij.T   -> (A,F)
            grad["dW_HP_L"] += np.einsum("nab,gb,nfg->af", delta_pre_h, net.W_HP_R, p_ij)
            # dW_HP_R = p_ij.T @ W_HP_L.T @ delta_pre_h   -> (F,B)
            grad["dW_HP_R"] += np.einsum("ngf,ag,nab->fb", p_ij, net.W_HP_L, delta_pre_h)
            # dW_H_L = delta_pre_h @ W_H_R.T @ h_prev.T   -> (A,A)
            grad["dW_H_L"] += np.einsum("nab,eb,nde->ad", delta_pre_h, net.W_H_R, h_prev)
            # dW_H_R = h_prev.T @ W_H_L.T @ delta_pre_h   -> (B,B)
            grad["dW_H_R"] += np.einsum("nab,ea,nec->bc", h_prev, net.W_H_L, delta_pre_h)
            grad["dB_H"] += delta_pre_h.sum(axis=0)

            # delta_h_new = W_H_L.T @ delta_pre_h @ W_H_R.T   -> (N,A,B)
            delta_h = np.einsum("ea,nec,bc->nab", net.W_H_L, delta_pre_h, net.W_H_R)
            # dloss_dp = W_HP_L.T @ delta_pre_h @ W_HP_R.T    -> (N,F,F)
            dloss_dp = np.einsum("af,nab,gb->nfg", net.W_HP_L, delta_pre_h, net.W_HP_R)

            pre_feat = np.einsum("fp,npq,qg->nfg", net.W_FeatL, x, net.W_FeatR) + net.B_Feat
            delta_pre_p = dloss_dp * (pre_feat > 0)                        # (N,F,F)

            # dW_FeatL = delta_pre_p @ W_FeatR.T @ x.T   -> (F,P)
            grad["dW_FeatL"] += np.einsum("nfg,qg,npq->fp", delta_pre_p, net.W_FeatR, x)
            # dW_FeatR = x.T @ W_FeatL.T @ delta_pre_p   -> (P,F)
            grad["dW_FeatR"] += np.einsum("nqp,gq,ngf->pf", x, net.W_FeatL, delta_pre_p)
            grad["dB_Feat"] += delta_pre_p.sum(axis=0)

    return grad


def margin_loss_batch(net: TwoDRNN, prob: np.ndarray, onehot: np.ndarray) -> float:
    gamma = net.loss["gamma"]; mp = net.loss["m_plus"]; mm = net.loss["m_minus"]
    total = (onehot * np.maximum(mp - prob, 0.0) ** 2
             + gamma * (1.0 - onehot) * np.maximum(prob - mm, 0.0) ** 2)
    return float(total.sum())
