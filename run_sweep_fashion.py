"""Fashion-MNIST replication of run_sweep.py / Table 5 (the f-sweep, 2D-RNN).
Same hyperparameters as the MNIST sweep (batch=1000, lr=0.01, init_scale=0.3,
100 epochs) -- only the dataset changes (drop-in replacement, same 28x28 shape).
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

RESULTS_DIR = Path("results_fashion")


def _result_path(f: int) -> Path:
    return RESULTS_DIR / f"f{f}.json"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Fashion-MNIST RNN sweep start: f={F_VALUES}, epochs={TOTAL_EPOCH}, batch={BATCH_SIZE}, "
          f"lr={LEARNING_RATE}, init_scale={INIT_SCALE}", flush=True)

    for f in F_VALUES:
        out_path = _result_path(f)
        if out_path.exists():
            print(f"[f={f}] already done, skipping ({out_path})", flush=True)
            continue

        print(f"\n===== Fashion-MNIST: Starting training for f={f} =====", flush=True)
        t0 = time.time()
        result = train_for_f(
            f, total_epoch=TOTAL_EPOCH, batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE, init_scale=INIT_SCALE,
            device="cuda", verbose=True, save=True, dataset="fashion_mnist",
        )
        elapsed = time.time() - t0
        print(f"f={f} finished in {elapsed:.1f}s | "
              f"train {result['train_accuracy'] * 100:.2f}% | "
              f"test {result['test_accuracy'] * 100:.2f}%", flush=True)

        out = {
            "f": int(result["f"]), "param_n": int(result["param_n"]),
            "train_loss": float(result["train_loss"]), "train_accuracy": float(result["train_accuracy"]),
            "test_accuracy": float(result["test_accuracy"]), "test_loss": float(result["test_loss"]),
            "run_time": float(result["run_time"]),
        }
        out_path.write_text(json.dumps(out), encoding="utf-8")

    print("\nAll f values done. Building final table...", flush=True)
    import build_table_fashion
    build_table_fashion.main()
    print("Fashion-MNIST RNN sweep complete.", flush=True)


if __name__ == "__main__":
    main()
