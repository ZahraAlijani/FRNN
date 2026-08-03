"""Fashion-MNIST replication of the dense 2D-NN-2L sweep (Table 2), same mu2
values as the paper's Tiny Ultra-Light category, 100 epochs, lr=0.001 (see
train_dense.py docstring for why we use a fixed lr instead of the paper's
eta=L_bar/3 rule)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from train_dense import train_dense

MU2_VALUES = [13, 17, 19, 23, 31, 41, 47, 53, 59, 67, 71]
TOTAL_EPOCH = 100

RESULTS_DIR = Path("results_dense_fashion")


def _result_path(mu2: int) -> Path:
    return RESULTS_DIR / f"mu2_{mu2}.json"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    print(f"Fashion-MNIST dense 2D-NN-2L sweep start: mu2={MU2_VALUES}, epochs={TOTAL_EPOCH}", flush=True)

    for mu2 in MU2_VALUES:
        out_path = _result_path(mu2)
        if out_path.exists():
            print(f"[mu2={mu2}] already done, skipping ({out_path})", flush=True)
            continue

        print(f"\n===== Fashion-MNIST dense: mu2={mu2} =====", flush=True)
        t0 = time.time()
        r = train_dense(mu2, total_epoch=TOTAL_EPOCH, device="cuda", dataset="fashion_mnist")
        elapsed = time.time() - t0
        print(f"mu2={mu2} finished in {elapsed:.1f}s | train {r['train_accuracy']*100:.2f}% | "
              f"test {r['test_accuracy']*100:.2f}%", flush=True)

        out = {k: v for k, v in r.items()}
        out_path.write_text(json.dumps(out), encoding="utf-8")

    print("\nAll mu2 values done. Building final table...", flush=True)
    import build_table_dense_fashion
    build_table_dense_fashion.main()
    print("Fashion-MNIST dense sweep complete.", flush=True)


if __name__ == "__main__":
    main()
