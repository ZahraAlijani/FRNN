from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TwoDRNN:
    W_FeatL: np.ndarray
    W_FeatR: np.ndarray
    B_Feat: np.ndarray
    W_HP_L: np.ndarray
    W_HP_R: np.ndarray
    W_H_L: np.ndarray
    W_H_R: np.ndarray
    B_H: np.ndarray
    W_InfL: np.ndarray
    W_InfR: np.ndarray
    B_Inf: np.ndarray
    loss: dict[str, float]
    adam: dict[str, Any]
    p: int
    f: int
    h1: int
    h2: int
    m_grid: int


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


def _zero_buffers_like(net: TwoDRNN) -> dict[str, np.ndarray]:
    return {
        "dW_FeatL": np.zeros_like(net.W_FeatL),
        "dW_FeatR": np.zeros_like(net.W_FeatR),
        "dB_Feat": np.zeros_like(net.B_Feat),
        "dW_HP_L": np.zeros_like(net.W_HP_L),
        "dW_HP_R": np.zeros_like(net.W_HP_R),
        "dW_H_L": np.zeros_like(net.W_H_L),
        "dW_H_R": np.zeros_like(net.W_H_R),
        "dB_H": np.zeros_like(net.B_H),
        "dW_InfL": np.zeros_like(net.W_InfL),
        "dW_InfR": np.zeros_like(net.W_InfR),
        "dB_Inf": np.zeros_like(net.B_Inf),
    }


def create_net_2drnn(
    p: int,
    f: int,
    h1: int,
    h2: int,
    m: int,
    seed: int | None = None,
) -> TwoDRNN:
    rng = np.random.default_rng(seed)
    scale_init = 0.01

    return TwoDRNN(
        W_FeatL=scale_init * rng.standard_normal((f, p)),
        W_FeatR=scale_init * rng.standard_normal((p, f)),
        B_Feat=np.zeros((f, f)),
        W_HP_L=scale_init * rng.standard_normal((h1, f)),
        W_HP_R=scale_init * rng.standard_normal((f, h2)),
        W_H_L=scale_init * rng.standard_normal((h1, h1)),
        W_H_R=scale_init * rng.standard_normal((h2, h2)),
        B_H=np.zeros((h1, h2)),
        W_InfL=scale_init * rng.standard_normal((10, h1)),
        W_InfR=scale_init * rng.standard_normal((h2, 1)),
        B_Inf=np.zeros((10, 1)),
        loss={"gamma": 1.0, "m_plus": 0.9, "m_minus": 0.1},
        adam={
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "t": 0,
            "m": None,
            "v": None,
        },
        p=p,
        f=f,
        h1=h1,
        h2=h2,
        m_grid=m,
    )


def forward_2drnn(
    net: TwoDRNN,
    img_input: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray, list[list[np.ndarray]], list[list[np.ndarray]]]:
    p = net.p
    m = net.m_grid

    patch_store: list[list[np.ndarray]] = [[None for _ in range(m)] for _ in range(m)]
    hidden_history: list[list[np.ndarray]] = [[None for _ in range(m)] for _ in range(m)]

    for i in range(m):
        for j in range(m):
            row_start = i * p
            row_end = (i + 1) * p
            col_start = j * p
            col_end = (j + 1) * p
            x_patch = img_input[row_start:row_end, col_start:col_end]
            z_pre = net.W_FeatL @ x_patch @ net.W_FeatR + net.B_Feat
            patch_store[i][j] = _relu(z_pre)

    h_prev = np.zeros((net.h1, net.h2))
    for i in range(m):
        for j in range(m):
            p_current = patch_store[i][j]
            pre_h = net.W_HP_L @ p_current @ net.W_HP_R + net.W_H_L @ h_prev @ net.W_H_R + net.B_H
            h_current = _sigmoid(pre_h)
            hidden_history[i][j] = h_current
            h_prev = h_current

    h_final = h_prev
    logit_vec = net.W_InfL @ h_final @ net.W_InfR + net.B_Inf
    prob_d = _softmax(logit_vec)
    pred_digit = int(np.argmax(prob_d))
    return prob_d, pred_digit, h_final, patch_store, hidden_history


def margin_loss(net: TwoDRNN, prob_d: np.ndarray, onehot_label: np.ndarray) -> float:
    gamma = net.loss["gamma"]
    m_plus = net.loss["m_plus"]
    m_minus = net.loss["m_minus"]
    total = 0.0
    for k in range(10):
        yk = onehot_label[k]
        dk = prob_d[k]
        total += yk * max(m_plus - dk, 0.0) ** 2 + gamma * (1.0 - yk) * max(dk - m_minus, 0.0) ** 2
    return float(total)


def backward_2drnn(
    net: TwoDRNN,
    img_input: np.ndarray,
    onehot_label: np.ndarray,
    prob_d: np.ndarray,
    h_final: np.ndarray,
    patch_store: list[list[np.ndarray]],
    hidden_history: list[list[np.ndarray]],
) -> dict[str, np.ndarray]:
    p = net.p
    h1 = net.h1
    h2 = net.h2
    m = net.m_grid

    gamma = net.loss["gamma"]
    m_plus = net.loss["m_plus"]
    m_minus = net.loss["m_minus"]

    grad = _zero_buffers_like(net)

    dL_dd = np.zeros((10, 1))
    for k in range(10):
        yk = float(onehot_label[k])
        dk = float(prob_d[k])
        dL_dd[k] = -2.0 * yk * max(m_plus - dk, 0.0) + 2.0 * gamma * (1.0 - yk) * max(dk - m_minus, 0.0)

    prob_flat = prob_d.reshape(10, 1)
    logit_grad = np.zeros((10, 1))
    for t_idx in range(10):
        total_sum = 0.0
        for k_idx in range(10):
            delta = 1.0 if k_idx == t_idx else 0.0
            total_sum += float(dL_dd[k_idx]) * float(prob_flat[k_idx]) * (delta - float(prob_flat[t_idx]))
        logit_grad[t_idx] = total_sum

    delta_inference = logit_grad
    grad["dW_InfL"] = delta_inference @ net.W_InfR.T @ h_final.T
    grad["dW_InfR"] = h_final.T @ net.W_InfL.T @ delta_inference
    grad["dB_Inf"] = delta_inference

    delta_current_h = net.W_InfL.T @ delta_inference @ net.W_InfR.T

    for i in range(m - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            h_ij = hidden_history[i][j]
            p_ij = patch_store[i][j]
            x_patch = img_input[i * p:(i + 1) * p, j * p:(j + 1) * p]

            if i == 0 and j == 0:
                h_prev = np.zeros((h1, h2))
            elif j == 0:
                h_prev = hidden_history[i - 1][m - 1]
            else:
                h_prev = hidden_history[i][j - 1]

            sigma_deriv = h_ij * (1.0 - h_ij)
            delta_pre_h = delta_current_h * sigma_deriv

            grad["dW_HP_L"] += delta_pre_h @ net.W_HP_R.T @ p_ij.T
            grad["dW_HP_R"] += p_ij.T @ net.W_HP_L.T @ delta_pre_h
            grad["dW_H_L"] += delta_pre_h @ net.W_H_R.T @ h_prev.T
            grad["dW_H_R"] += h_prev.T @ net.W_H_L.T @ delta_pre_h
            grad["dB_H"] += delta_pre_h

            delta_current_h = net.W_H_L.T @ delta_pre_h @ net.W_H_R.T
            dloss_d_pij = net.W_HP_L.T @ delta_pre_h @ net.W_HP_R.T

            pre_p_feature = net.W_FeatL @ x_patch @ net.W_FeatR + net.B_Feat
            relu_mask = pre_p_feature > 0
            delta_pre_p = dloss_d_pij * relu_mask

            grad["dW_FeatL"] += delta_pre_p @ net.W_FeatR.T @ x_patch.T
            grad["dW_FeatR"] += x_patch.T @ net.W_FeatL.T @ delta_pre_p
            grad["dB_Feat"] += delta_pre_p

    return grad


def adam_update_2drnn(net: TwoDRNN, grad_struct: dict[str, np.ndarray], learning_rate: float) -> TwoDRNN:
    net.adam["t"] += 1
    t = net.adam["t"]
    beta1 = net.adam["beta1"]
    beta2 = net.adam["beta2"]
    eps = net.adam["eps"]

    if net.adam["m"] is None:
        net.adam["m"] = _zero_buffers_like(net)
        net.adam["v"] = _zero_buffers_like(net)

    moments = net.adam["m"]
    variances = net.adam["v"]

    def update(param_name: str, grad_name: str) -> None:
        g = grad_struct[grad_name]
        moments[grad_name] = beta1 * moments[grad_name] + (1.0 - beta1) * g
        variances[grad_name] = beta2 * variances[grad_name] + (1.0 - beta2) * (g ** 2)
        m_hat = moments[grad_name] / (1.0 - beta1 ** t)
        v_hat = variances[grad_name] / (1.0 - beta2 ** t)
        setattr(net, param_name, getattr(net, param_name) - learning_rate * m_hat / (np.sqrt(v_hat) + eps))

    update("W_FeatL", "dW_FeatL")
    update("W_FeatR", "dW_FeatR")
    update("B_Feat", "dB_Feat")
    update("W_HP_L", "dW_HP_L")
    update("W_HP_R", "dW_HP_R")
    update("W_H_L", "dW_H_L")
    update("W_H_R", "dW_H_R")
    update("B_H", "dB_H")
    update("W_InfL", "dW_InfL")
    update("W_InfR", "dW_InfR")
    update("B_Inf", "dB_Inf")
    return net


def one_hot_label(label: int, num_classes: int = 10) -> np.ndarray:
    vec = np.zeros((num_classes, 1))
    vec[int(label)] = 1.0
    return vec


def matlab_label_to_digit(labels: np.ndarray) -> np.ndarray:
    flat = np.asarray(labels).squeeze()
    if np.max(flat) <= 1.0:
        flat = flat * 10
    return flat.astype(int)


create_net_2DRNN = create_net_2drnn
forward_2DRNN = forward_2drnn
backward_2DRNN = backward_2drnn
adam_update_2DRNN = adam_update_2drnn
