"""Fast (vectorized minibatch) training of the 2D-RNN.

Same model and math as ``train_example.py`` / ``TrainExample.m``, but the whole
minibatch is processed at once with the batched kernels in ``two_drnn_fast.py``
and a single Adam step is taken per minibatch (mean gradient). This is 1-2 orders
of magnitude faster than the original one-sample-at-a-time online loop.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.io import loadmat, savemat

from two_drnn import create_net_2drnn, adam_update_2drnn, matlab_label_to_digit
import two_drnn_fast as F

_CACHE: dict[str, object] = {}


def _load_data():
    """Load and cache MNIST as (N,28,28) float arrays plus integer labels."""
    if "x_train" not in _CACHE:
        tr = loadmat("MNIST2.mat")
        x_train = np.ascontiguousarray(np.transpose(np.asarray(tr["MNSIT_Train"]), (2, 0, 1)))
        y_train = matlab_label_to_digit(tr["MNSIT_TrainL"]).astype(int)
        te = loadmat("MNIST2Test.mat")
        x_test = np.ascontiguousarray(np.transpose(np.asarray(te["X_Test"]), (2, 0, 1)))
        y_test = matlab_label_to_digit(te["Y_Test"]).astype(int)
        _CACHE.update(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return _CACHE["x_train"], _CACHE["y_train"], _CACHE["x_test"], _CACHE["y_test"]


def _evaluate(net, x, y, eval_batch=2000):
    correct = 0
    loss = 0.0
    n = x.shape[0]
    for s in range(0, n, eval_batch):
        xb = x[s:s + eval_batch]
        yb = y[s:s + eval_batch]
        onehot = np.zeros((xb.shape[0], 10, 1))
        onehot[np.arange(xb.shape[0]), yb, 0] = 1.0
        prob, _, _, _ = F.forward_batch(net, xb)
        loss += F.margin_loss_batch(net, prob, onehot)
        correct += int((np.argmax(prob[:, :, 0], axis=1) == yb).sum())
    return correct / n, loss / n


_WEIGHT_ATTRS = ("W_FeatL", "W_FeatR", "W_HP_L", "W_HP_R",
                 "W_H_L", "W_H_R", "W_InfL", "W_InfR")


def train_for_f(f, total_epoch=50, batch_size=100, learning_rate=0.01,
                init_scale=0.3, seed=0, verbose=True, save=True):
    p, h1, h2, m_grid = 7, 10, 1, 4
    x_train, y_train, x_test, y_test = _load_data()
    n_train = x_train.shape[0]

    net = create_net_2drnn(p, f, h1, h2, m_grid, seed=seed)
    # Rescale the (tiny 0.01) default init. At 0.01 the network output is nearly
    # input-independent, so a minibatch-averaged gradient ~cancels and training
    # stalls at a symmetric saddle. A larger init makes it input-sensitive and
    # lets minibatch training escape immediately.
    if init_scale != 0.01:
        for attr in _WEIGHT_ATTRS:
            setattr(net, attr, getattr(net, attr) * (init_scale / 0.01))
    rng = np.random.default_rng(seed)

    loss_log = np.zeros(total_epoch)
    accuracy_log = np.zeros(total_epoch)

    start = time.time()
    for epoch in range(total_epoch):
        order = rng.permutation(n_train)
        xs = x_train[order]
        ys = y_train[order]

        cum_loss = 0.0
        correct = 0
        for b in range(0, n_train, batch_size):
            xb = xs[b:b + batch_size]
            yb = ys[b:b + batch_size]
            onehot = np.zeros((xb.shape[0], 10, 1))
            onehot[np.arange(xb.shape[0]), yb, 0] = 1.0

            prob, h_final, patch_store, hidden_history = F.forward_batch(net, xb)
            cum_loss += F.margin_loss_batch(net, prob, onehot)
            correct += int((np.argmax(prob[:, :, 0], axis=1) == yb).sum())

            grad = F.backward_batch(net, xb, onehot, prob, h_final, patch_store, hidden_history)
            for k in grad:  # mean gradient over the minibatch
                grad[k] /= xb.shape[0]
            net = adam_update_2drnn(net, grad, learning_rate)

        loss_log[epoch] = cum_loss / n_train
        accuracy_log[epoch] = correct / n_train
        if verbose:
            print(f"f={f} | Epoch {epoch + 1}/{total_epoch} | Loss {loss_log[epoch]:.4f} | "
                  f"TrainAcc {accuracy_log[epoch]:.4f}")

    run_time = (time.time() - start) / total_epoch

    test_acc, test_loss = _evaluate(net, x_test, y_test)
    train_acc = accuracy_log[-1]
    train_loss = loss_log[-1]
    param_n = ((f * p) + (p * f) + (f * f) + (h1 * f) + (f * h2) + (h1 * h1)
               + (h2 * h2) + (h1 * h2) + (10 * h1) + (h2 * 1) + (10 * 1))

    if save:
        net_dict = {a: getattr(net, a) for a in _WEIGHT_ATTRS}
        net_dict.update(B_Feat=net.B_Feat, B_H=net.B_H, B_Inf=net.B_Inf)
        savemat(f"RNN2D_H({h1}t{h2})m{m_grid}f{f}.mat",
                {"net": net_dict, "loss_log": loss_log, "accuracy_log": accuracy_log,
                 "epoch_count": total_epoch})

    return {
        "f": f, "param_n": param_n, "train_loss": train_loss, "train_accuracy": train_acc,
        "test_accuracy": test_acc, "test_loss": test_loss, "run_time": run_time,
        "loss_log": loss_log, "accuracy_log": accuracy_log,
    }
