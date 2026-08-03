"""Train the MatNet baseline (matnet_torch.py) on MNIST under our own protocol
(Adam, GPU), for direct comparison against our 2D-NN/2D-RNN tables at matched
parameter counts. See matnet_torch.py docstring for the primary-source
equations and the two reference points we compare against (their reported
5,536-param/97.3% baseline and 316,160-param/98.48% best variant).
"""
from __future__ import annotations

import time

import torch

from matnet_torch import create_matnet, forward_batch, param_count
from train_torch import _load_data


@torch.no_grad()
def _evaluate(net, x, y, eval_batch=2000):
    correct = 0
    loss_sum = 0.0
    n = x.shape[0]
    for s in range(0, n, eval_batch):
        xb = x[s:s + eval_batch]
        yb = y[s:s + eval_batch]
        prob = forward_batch(net, xb)
        loss_sum += -torch.log(prob[torch.arange(xb.shape[0]), yb] + 1e-12).sum().item()
        correct += int((prob.argmax(dim=1) == yb).sum().item())
    return correct / n, loss_sum / n


def train_matnet(hidden_dims, total_epoch=100, batch_size=1000, learning_rate=0.001,
                  seed=0, device="cuda", verbose=True):
    dtype = torch.float64
    x_train, y_train, x_test, y_test = _load_data(device, dtype)
    n_train = x_train.shape[0]

    net = create_matnet(hidden_dims, in_dim=28, seed=seed, device=device, dtype=dtype)
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

            prob = forward_batch(net, xb)
            loss = -torch.log(prob[torch.arange(xb.shape[0]), yb] + 1e-12).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item() * xb.shape[0]
            correct += int((prob.argmax(dim=1) == yb).sum().item())

        train_loss = cum_loss / (n_batches * batch_size)
        train_acc = correct / (n_batches * batch_size)
        if verbose:
            print(f"MatNet{hidden_dims} | Epoch {epoch + 1}/{total_epoch} | "
                  f"Loss {train_loss:.4f} | TrainAcc {train_acc:.4f}", flush=True)

    run_time = (time.time() - start) / total_epoch
    test_acc, test_loss = _evaluate(net, x_test, y_test)
    n_params = param_count(hidden_dims)

    return {
        "hidden_dims": hidden_dims, "param_n": n_params, "train_loss": train_loss,
        "train_accuracy": train_acc, "test_accuracy": test_acc, "test_loss": test_loss,
        "run_time": run_time,
    }


if __name__ == "__main__":
    import json
    from pathlib import Path

    Path("results_matnet").mkdir(exist_ok=True)
    configs = [[20, 16], [40, 32], [80, 64], [160, 160]]
    for dims in configs:
        tag = "x".join(str(d) for d in dims)
        out_path = Path(f"results_matnet/matnet_{tag}.json")
        if out_path.exists():
            print(f"[{dims}] already done, skipping", flush=True)
            continue
        print(f"\n===== MatNet {dims} =====", flush=True)
        r = train_matnet(dims, total_epoch=100, batch_size=1000, learning_rate=0.001, device="cuda")
        out = {k: v for k, v in r.items() if k != "hidden_dims"}
        out["hidden_dims"] = dims
        out_path.write_text(json.dumps(out), encoding="utf-8")
        print(f"[{dims}] params={r['param_n']} train_acc={r['train_accuracy']*100:.2f}% "
              f"test_acc={r['test_accuracy']*100:.2f}% T={r['run_time']:.2f}s/epoch", flush=True)
    print("\nAll MatNet configs done.", flush=True)
