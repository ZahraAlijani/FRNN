"""Combine per-f result JSONs into the summary table + LaTeX table."""
from __future__ import annotations

import glob
import json
from pathlib import Path

TOTAL_EPOCH = 100


def main() -> None:
    rows = []
    for path in glob.glob("results/f*.json"):
        rows.append(json.loads(Path(path).read_text(encoding="utf-8")))
    rows.sort(key=lambda r: r["f"])

    print(f"\nTable 4. Training parameters and performance (Batch=100, {TOTAL_EPOCH} epochs)")
    print(f"{'f':>3} {'Num.N':>7} {'Loss':>8} {'TrainAcc%':>10} {'TestAcc%':>9} {'T(s/ep)':>8}")
    for r in rows:
        print(f"{r['f']:>3} {r['param_n']:>7} {r['train_loss']:>8.4f} "
              f"{r['train_accuracy']*100:>10.4f} {r['test_accuracy']*100:>9.4f} {r['run_time']:>8.4f}")

    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Training parameters and performance. Batch size = 100, Epoch = {TOTAL_EPOCH}.}}",
        r"\label{tab:training_parameters_performance}",
        r"\begin{tabular}{r r r r r r r}", r"\hline",
        r"Epoch & f & Num. Neuron & Loss & Train acc & Test acc & T(s) \\", r"\hline",
    ]
    for r in rows:
        lines.append(f"{TOTAL_EPOCH} & {r['f']} & {r['param_n']} & {r['train_loss']:.4f} & "
                     f"{r['train_accuracy']*100:.4f} & {r['test_accuracy']*100:.4f} & {r['run_time']:.4f} \\\\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    Path("table4_training_report.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved LaTeX table to table4_training_report.tex ({len(rows)} rows)")


if __name__ == "__main__":
    main()
