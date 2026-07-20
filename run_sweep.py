"""Run the full f-sweep sequentially on GPU, writing per-f JSON results as it goes
(so partial progress survives a crash/disconnect) plus the final LaTeX table + CSV.

Intended to be launched inside tmux (see instruction.md) so it survives SSH
disconnects:
    tmux new -d -s frnn_sweep 'conda run -n env1 --no-capture-output python run_sweep.py'
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from train_torch import train_for_f

F_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
TOTAL_EPOCH = 100
LEARNING_RATE = 0.01
INIT_SCALE = 0.3
BATCH_SIZE = 1000

RESULTS_DIR = Path("results")


def _result_path(f: int) -> Path:
    return RESULTS_DIR / f"f{f}.json"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Sweep start: f={F_VALUES}, epochs={TOTAL_EPOCH}, batch={BATCH_SIZE}, "
          f"lr={LEARNING_RATE}, init_scale={INIT_SCALE}", flush=True)

    for f in F_VALUES:
        out_path = _result_path(f)
        if out_path.exists():
            print(f"[f={f}] already done, skipping ({out_path})", flush=True)
            continue

        print(f"\n===== Starting training for f={f} =====", flush=True)
        t0 = time.time()
        result = train_for_f(
            f, total_epoch=TOTAL_EPOCH, batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE, init_scale=INIT_SCALE,
            device="cuda", verbose=True, save=True,
        )
        elapsed = time.time() - t0
        print(f"f={f} finished in {elapsed:.1f}s | "
              f"train {result['train_accuracy'] * 100:.2f}% | "
              f"test {result['test_accuracy'] * 100:.2f}%", flush=True)

        out = {
            "f": int(result["f"]),
            "param_n": int(result["param_n"]),
            "train_loss": float(result["train_loss"]),
            "train_accuracy": float(result["train_accuracy"]),
            "test_accuracy": float(result["test_accuracy"]),
            "test_loss": float(result["test_loss"]),
            "run_time": float(result["run_time"]),
        }
        out_path.write_text(json.dumps(out), encoding="utf-8")

    print("\nAll f values done. Building final table...", flush=True)
    import build_table
    build_table.main()
    print("Sweep complete.", flush=True)


if __name__ == "__main__":
    main()
