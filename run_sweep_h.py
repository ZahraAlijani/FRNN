"""Hidden-state sweep: h1=h2=h for h in H_VALUES, with f fixed (see instruction.md
follow-up: "Hidden state variables H" example, f held at the best f from the
f-sweep). Mirrors run_sweep.py's structure -- per-h JSON checkpoints so partial
progress survives a crash/disconnect, launched inside tmux for the same reason.

    tmux new -d -s frnn_sweep_h 'conda run -n env1 --no-capture-output python run_sweep_h.py'
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from train_torch import train_for_f

FIXED_F = 20
H_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
TOTAL_EPOCH = 100
LEARNING_RATE = 0.01
INIT_SCALE = 0.3
BATCH_SIZE = 1000

RESULTS_DIR = Path("results_h")


def _result_path(h: int) -> Path:
    return RESULTS_DIR / f"h{h}.json"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"H-sweep start: f={FIXED_F} (fixed), h1=h2={H_VALUES}, epochs={TOTAL_EPOCH}, "
          f"batch={BATCH_SIZE}, lr={LEARNING_RATE}, init_scale={INIT_SCALE}", flush=True)

    for h in H_VALUES:
        out_path = _result_path(h)
        if out_path.exists():
            print(f"[h={h}] already done, skipping ({out_path})", flush=True)
            continue

        print(f"\n===== Starting training for h1=h2={h} (f={FIXED_F}) =====", flush=True)
        t0 = time.time()
        result = train_for_f(
            FIXED_F, total_epoch=TOTAL_EPOCH, batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE, init_scale=INIT_SCALE,
            h1=h, h2=h, device="cuda", verbose=True, save=True,
        )
        elapsed = time.time() - t0
        print(f"h={h} finished in {elapsed:.1f}s | "
              f"train {result['train_accuracy'] * 100:.2f}% | "
              f"test {result['test_accuracy'] * 100:.2f}%", flush=True)

        out = {
            "h": h,
            "f": FIXED_F,
            "param_n": int(result["param_n"]),
            "train_loss": float(result["train_loss"]),
            "train_accuracy": float(result["train_accuracy"]),
            "test_accuracy": float(result["test_accuracy"]),
            "test_loss": float(result["test_loss"]),
            "run_time": float(result["run_time"]),
        }
        out_path.write_text(json.dumps(out), encoding="utf-8")

    print("\nAll h values done. Building final table...", flush=True)
    import build_table_h
    build_table_h.main()
    print("H-sweep complete.", flush=True)


if __name__ == "__main__":
    main()
