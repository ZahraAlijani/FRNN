"""Standard batched (minibatch) GPU training of the 2D-RNN using PyTorch.

Model equations match MATLAB exactly (see two_drnn_torch.py, verified against
matlab.mat to 1e-18). Training procedure is standard minibatch Adam -- one
gradient (averaged over the batch) and one optimizer step per batch -- rather
than MATLAB's literal per-sample online update loop (TrainExample.m calls
adam_update_2DRNN once per single image, 60000 times/epoch; see chat history).
That per-sample loop is inherently sequential and has no batch parallelism, so
it cannot be accelerated on GPU. Batched training is the standard/GPU-friendly
alternative, using the same hyperparameters MATLAB used (batch_size=1000,
lr=0.001, init_scale=0.01) as the starting point.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.io import loadmat, savemat

from two_drnn import matlab_label_to_digit
from two_drnn_torch import create_net_torch, forward_batch, margin_loss_batch

_CACHE: dict[str, object] = {}


def _load_data(device: str, dtype: torch.dtype):
    if "x_train" not in _CACHE:
        tr = loadmat("MNIST2.mat")
        x_train = np.ascontiguousarray(np.transpose(np.asarray(tr["MNSIT_Train"]), (2, 0, 1))).astype(np.float64)
        y_train = matlab_label_to_digit(tr["MNSIT_TrainL"]).astype(np.int64)
        te = loadmat("MNIST2Test.mat")
        x_test = np.ascontiguousarray(np.transpose(np.asarray(te["X_Test"]), (2, 0, 1))).astype(np.float64)
        y_test = matlab_label_to_digit(te["Y_Test"]).astype(np.int64)
        _CACHE.update(
            x_train=torch.tensor(x_train, dtype=dtype, device=device),
            y_train=torch.tensor(y_train, device=device),
            x_test=torch.tensor(x_test, dtype=dtype, device=device),
            y_test=torch.tensor(y_test, device=device),
        )
    return _CACHE["x_train"], _CACHE["y_train"], _CACHE["x_test"], _CACHE["y_test"]


_WEIGHT_ATTRS = ("W_FeatL", "W_FeatR", "W_HP_L", "W_HP_R",
                  "W_H_L", "W_H_R", "W_InfL", "W_InfR")
_BIAS_ATTRS = ("B_Feat", "B_H", "B_Inf")


@torch.no_grad()
def _evaluate(net, x, y, eval_batch=2000):
    correct = 0
    loss_sum = 0.0
    n = x.shape[0]
    for s in range(0, n, eval_batch):
        xb = x[s:s + eval_batch]
        yb = y[s:s + eval_batch]
        onehot = torch.zeros(xb.shape[0], 10, 1, dtype=x.dtype, device=x.device)
        onehot[torch.arange(xb.shape[0]), yb, 0] = 1.0
        prob, _ = forward_batch(net, xb)
        loss_sum += margin_loss_batch(prob, onehot).item() * xb.shape[0]
        correct += int((prob[:, :, 0].argmax(dim=1) == yb).sum().item())
    return correct / n, loss_sum / n


def train_for_f(f, total_epoch=100, batch_size=1000, learning_rate=0.001,
                 init_scale=0.01, seed=0, device="cuda", verbose=True, save=True,
                 h1=10, h2=1, p=7, m_grid=4):
    dtype = torch.float64
    x_train, y_train, x_test, y_test = _load_data(device, dtype)
    n_train = x_train.shape[0]

    net = create_net_torch(p, f, h1, h2, m_grid, seed=seed, device=device, dtype=dtype)
    if init_scale != 0.01:
        with torch.no_grad():
            for attr in _WEIGHT_ATTRS:
                getattr(net, attr).mul_(init_scale / 0.01)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    loss_log = np.zeros(total_epoch)
    accuracy_log = np.zeros(total_epoch)

    start = time.time()
    n_batches = n_train // batch_size
    for epoch in range(total_epoch):
        order = torch.randperm(n_train, generator=gen, device=device)
        xs = x_train[order]
        ys = y_train[order]

        cum_loss = 0.0
        correct = 0
        for b in range(n_batches):
            xb = xs[b * batch_size:(b + 1) * batch_size]
            yb = ys[b * batch_size:(b + 1) * batch_size]
            onehot = torch.zeros(xb.shape[0], 10, 1, dtype=dtype, device=device)
            onehot[torch.arange(xb.shape[0]), yb, 0] = 1.0

            prob, h_final = forward_batch(net, xb)
            loss = margin_loss_batch(prob, onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item() * xb.shape[0]
            correct += int((prob[:, :, 0].argmax(dim=1) == yb).sum().item())

        loss_log[epoch] = cum_loss / (n_batches * batch_size)
        accuracy_log[epoch] = correct / (n_batches * batch_size)
        if verbose:
            print(f"f={f} h1={h1} h2={h2} | Epoch {epoch + 1}/{total_epoch} | Loss {loss_log[epoch]:.4f} | "
                  f"TrainAcc {accuracy_log[epoch]:.4f}", flush=True)

    run_time = (time.time() - start) / total_epoch

    test_acc, test_loss = _evaluate(net, x_test, y_test)
    train_acc = accuracy_log[-1]
    train_loss = loss_log[-1]
    param_n = ((f * p) + (p * f) + (f * f) + (h1 * f) + (f * h2) + (h1 * h1)
               + (h2 * h2) + (h1 * h2) + (10 * h1) + (h2 * 1) + (10 * 1))

    if save:
        net_dict = {a: getattr(net, a).detach().cpu().numpy() for a in _WEIGHT_ATTRS + _BIAS_ATTRS}
        savemat(f"RNN2D_H({h1}t{h2})m{m_grid}f{f}.mat",
                {"net": net_dict, "loss_log": loss_log, "accuracy_log": accuracy_log,
                 "epoch_count": total_epoch})

    return {
        "f": f, "param_n": param_n, "train_loss": train_loss, "train_accuracy": train_acc,
        "test_accuracy": test_acc, "test_loss": test_loss, "run_time": run_time,
        "loss_log": loss_log, "accuracy_log": accuracy_log,
    }
