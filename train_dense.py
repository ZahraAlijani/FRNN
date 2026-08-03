"""Train the dense 2D-NN-2L model (two_dense_torch.py) on MNIST, following the
paper's stated recipe (Xavier init, eta = mean_initial_loss / 3, Adam, 100
epochs, batch=1000/60 batches) so results are checkable against the existing
Table 2/3 rows in TwoD_NN.tex before trusting this implementation for
Fashion-MNIST.
"""
from __future__ import annotations

import time

import torch

from two_dense_torch import create_dense_net, forward_batch, margin_loss_batch, param_count
from train_torch import _load_data


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
        prob = forward_batch(net, xb)
        loss_sum += margin_loss_batch(prob, onehot).item() * xb.shape[0]
        correct += int((prob[:, :, 0].argmax(dim=1) == yb).sum().item())
    return correct / n, loss_sum / n


def train_dense(mu2, total_epoch=100, batch_size=1000, seed=0, device="cuda",
                 dataset="mnist", verbose=True, learning_rate=0.001):
    """NOTE: the paper text (TwoD_NN.tex) describes the initial learning rate as
    eta = mean_initial_loss / 3. We found that literal rule destroys training
    with Adam (computed lr ~0.21, which collapses all ReLUs dead within the
    first epoch -- confirmed reproducibly on mu2=41 and mu2=71, and confirmed
    NOT a forward/backward bug since the same architecture trains cleanly with
    a standard lr). Flagged for the paper authors to double check; we use a
    standard fixed lr here instead so results are actually reproducible."""
    dtype = torch.float64
    x_train, y_train, x_test, y_test = _load_data(device, dtype, dataset=dataset)
    n_train = x_train.shape[0]

    net = create_dense_net(mu2, in_dim=x_train.shape[1], seed=seed, device=device, dtype=dtype)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    n_batches = n_train // batch_size
    start = time.time()
    for epoch in range(total_epoch):
        order = torch.randperm(n_train, generator=gen, device=device)
        xs, ys = x_train[order], y_train[order]
        cum_loss, correct = 0.0, 0
        for b in range(n_batches):
            xb = xs[b * batch_size:(b + 1) * batch_size]
            yb = ys[b * batch_size:(b + 1) * batch_size]
            onehot = torch.zeros(xb.shape[0], 10, 1, dtype=dtype, device=device)
            onehot[torch.arange(xb.shape[0]), yb, 0] = 1.0

            prob = forward_batch(net, xb)
            loss = margin_loss_batch(prob, onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item() * xb.shape[0]
            correct += int((prob[:, :, 0].argmax(dim=1) == yb).sum().item())

        train_loss = cum_loss / (n_batches * batch_size)
        train_acc = correct / (n_batches * batch_size)
        if verbose:
            print(f"Dense2L mu2={mu2} | Epoch {epoch + 1}/{total_epoch} | "
                  f"Loss {train_loss:.4f} | TrainAcc {train_acc:.4f}", flush=True)

    run_time = (time.time() - start) / total_epoch
    test_acc, test_loss = _evaluate(net, x_test, y_test)

    return {
        "mu2": mu2, "param_n": param_count(mu2), "train_loss": train_loss,
        "train_accuracy": train_acc, "test_accuracy": test_acc, "test_loss": test_loss,
        "run_time": run_time, "learning_rate": learning_rate,
    }


if __name__ == "__main__":
    print("Verification run: mu2=41 (paper: 4438 params, train 98.85%, test 96.76%) "
          "and mu2=71 (paper: 9808 params, train 99.46%, test 97.48%)", flush=True)
    for mu2 in (41, 71):
        r = train_dense(mu2, total_epoch=100, device="cuda")
        print(f"[mu2={mu2}] params={r['param_n']} lr={r['learning_rate']:.5f} "
              f"train_acc={r['train_accuracy']*100:.2f}% test_acc={r['test_accuracy']*100:.2f}% "
              f"T={r['run_time']:.3f}s/epoch  (paper: see docstring)", flush=True)
