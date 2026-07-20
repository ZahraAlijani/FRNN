"""Train a single f value and write its result as JSON. Used for parallel sweep."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from train_fast import train_for_f


def main() -> None:
    f = int(sys.argv[1])
    total_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    result = train_for_f(f, total_epoch=total_epoch, verbose=True, save=True)
    out = {
        "f": int(result["f"]),
        "param_n": int(result["param_n"]),
        "train_loss": float(result["train_loss"]),
        "train_accuracy": float(result["train_accuracy"]),
        "test_accuracy": float(result["test_accuracy"]),
        "test_loss": float(result["test_loss"]),
        "run_time": float(result["run_time"]),
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/f{f}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"[f={f}] DONE test={out['test_accuracy']*100:.2f}% train={out['train_accuracy']*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
