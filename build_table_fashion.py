"""Combine per-f result JSONs (results_fashion/f*.json) into the LaTeX table + CSV
for the Fashion-MNIST 2D-RNN sweep (generalization check for Table 5)."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

TOTAL_EPOCH = 100
BATCH_SIZE = 1000
N_BATCHES = 60


def _build_latex_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Model performance on Fashion-MNIST under varying feature dimension $f$ "
        rf"(2D-RNN, $m=4$, $p=7$, $h_1=10$, $h_2=1$). "
        rf"Batch size = {BATCH_SIZE}, {N_BATCHES} batches, {TOTAL_EPOCH} epochs.}}",
        r"\label{tab:fashion_mnist_f_sweep}",
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"$f$ & Num. Neuron & Loss    & Train acc & Test acc & $T$ (s) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['f']}      & {r['param_n']}  & {r['train_loss']:.4f} & "
            f"{r['train_accuracy'] * 100:.4f} &{r['test_accuracy'] * 100:.4f} & "
            f"{r['run_time']:.4f}  \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    rows = []
    for path in glob.glob("results_fashion/f*.json"):
        rows.append(json.loads(Path(path).read_text(encoding="utf-8")))
    rows.sort(key=lambda r: r["f"])

    print(f"\nTable. Fashion-MNIST RNN f-sweep (Batch={BATCH_SIZE}, {TOTAL_EPOCH} epochs)")
    print(f"{'f':>3} {'Num.N':>7} {'Loss':>8} {'TrainAcc%':>10} {'TestAcc%':>9} {'T(s/ep)':>8}")
    for r in rows:
        print(f"{r['f']:>3} {r['param_n']:>7} {r['train_loss']:>8.4f} "
              f"{r['train_accuracy']*100:>10.4f} {r['test_accuracy']*100:>9.4f} {r['run_time']:>8.4f}")

    Path("table_fashion_rnn_report.tex").write_text(_build_latex_table(rows), encoding="utf-8")
    print(f"\nSaved LaTeX table to table_fashion_rnn_report.tex ({len(rows)} rows)")

    csv_path = Path("table_fashion_rnn_report.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["f", "num_neuron", "loss", "train_acc_pct", "test_acc_pct", "T_s_per_epoch"])
        for r in rows:
            writer.writerow([
                r["f"], r["param_n"], f"{r['train_loss']:.6f}",
                f"{r['train_accuracy'] * 100:.4f}", f"{r['test_accuracy'] * 100:.4f}",
                f"{r['run_time']:.4f}",
            ])
    print(f"Saved CSV to {csv_path}")


if __name__ == "__main__":
    main()
