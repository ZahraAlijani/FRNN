"""Run a single custom-hyperparameter training (not part of the f=2..20 sweep).

Usage: python run_custom.py <f> <p> <m_grid> <h1> <h2> <learning_rate> <batch_size> [total_epoch]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from train_torch import train_for_f


def main() -> None:
    f = int(sys.argv[1])
    p = int(sys.argv[2])
    m_grid = int(sys.argv[3])
    h1 = int(sys.argv[4])
    h2 = int(sys.argv[5])
    learning_rate = float(sys.argv[6])
    batch_size = int(sys.argv[7])
    total_epoch = int(sys.argv[8]) if len(sys.argv) > 8 else 100

    print(f"Custom run: f={f} p={p} m_grid={m_grid} h1={h1} h2={h2} "
          f"lr={learning_rate} batch={batch_size} epochs={total_epoch}", flush=True)

    result = train_for_f(
        f, total_epoch=total_epoch, batch_size=batch_size, learning_rate=learning_rate,
        p=p, m_grid=m_grid, h1=h1, h2=h2, device="cuda", verbose=True, save=True,
    )

    Path("results").mkdir(exist_ok=True)
    out = {
        "f": f, "p": p, "m_grid": m_grid, "h1": h1, "h2": h2,
        "learning_rate": learning_rate, "batch_size": batch_size, "total_epoch": total_epoch,
        "param_n": int(result["param_n"]), "train_loss": float(result["train_loss"]),
        "train_accuracy": float(result["train_accuracy"]), "test_accuracy": float(result["test_accuracy"]),
        "test_loss": float(result["test_loss"]), "run_time": float(result["run_time"]),
    }
    tag = f"f{f}_p{p}_m{m_grid}_h{h1}t{h2}"
    json_path = Path(f"results/custom_{tag}.json")
    json_path.write_text(json.dumps(out), encoding="utf-8")
    print(f"Saved {json_path}", flush=True)

    csv_path = Path("custom_results.csv")
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["f", "p", "m_grid", "h1", "h2", "learning_rate", "batch_size",
                              "total_epoch", "num_neuron", "loss", "train_acc_pct", "test_acc_pct",
                              "T_s_per_epoch"])
        writer.writerow([f, p, m_grid, h1, h2, learning_rate, batch_size, total_epoch,
                          out["param_n"], f"{out['train_loss']:.6f}",
                          f"{out['train_accuracy']*100:.4f}", f"{out['test_accuracy']*100:.4f}",
                          f"{out['run_time']:.4f}"])
    print(f"Appended to {csv_path}", flush=True)

    print(f"\nf={f} | Num. N: {out['param_n']} | Loss (Tr) {out['train_loss']:.4f} | "
          f"Train acc {out['train_accuracy']*100:.4f} | Test acc {out['test_accuracy']*100:.4f} | "
          f"T={out['run_time']:.4f} (s)")
    print(f"Copy for Table:\n{f}      & {out['param_n']}  & {out['train_loss']:.4f} & "
          f"{out['train_accuracy']*100:.4f} &{out['test_accuracy']*100:.4f} & {out['run_time']:.4f}  \\\\")


if __name__ == "__main__":
    main()
