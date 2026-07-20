from __future__ import annotations

import time
from pathlib import Path

from train_fast import train_for_f

TOTAL_EPOCH = 50


def _format_table_row(row: dict) -> str:
    return (
        f"{TOTAL_EPOCH} & {row['f']} & {row['param_n']} & {row['train_loss']:.4f} & "
        f"{row['train_accuracy'] * 100:.4f} & {row['test_accuracy'] * 100:.4f} & {row['run_time']:.4f} \\\\\n"
    )


def _build_latex_table(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Training parameters and performance. Batch size = 100, Epoch = {TOTAL_EPOCH}.}}",
        r"\label{tab:training_parameters_performance}",
        r"\begin{tabular}{r r r r r r r}",
        r"\hline",
        r"Epoch & f & Num. Neuron & Loss & Train acc & Test acc & T(s) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(_format_table_row(row).rstrip())
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    rows = []
    for f in range(2, 21, 2):
        print(f"\n===== Starting training for f={f} =====", flush=True)
        t0 = time.time()
        result = train_for_f(f, total_epoch=TOTAL_EPOCH)
        print(f"f={f} finished in {time.time() - t0:.1f}s | "
              f"train {result['train_accuracy'] * 100:.2f}% | "
              f"test {result['test_accuracy'] * 100:.2f}%", flush=True)
        rows.append(result)

    output_path = Path("table4_training_report.tex")
    output_path.write_text(_build_latex_table(rows), encoding="utf-8")

    print(f"\nTable 4. Training parameters and performance (Batch=100, {TOTAL_EPOCH} epochs)")
    print("f -- Num. Neuron -- Loss -- Train acc -- Test acc -- T(s)")
    for row in rows:
        print(
            f"{row['f']} -- {row['param_n']} -- {row['train_loss']:.4f} -- "
            f"{row['train_accuracy'] * 100:.4f} -- {row['test_accuracy'] * 100:.4f} -- {row['run_time']:.4f}"
        )
    print(f"\nSaved LaTeX table to {output_path}")


if __name__ == "__main__":
    main()
