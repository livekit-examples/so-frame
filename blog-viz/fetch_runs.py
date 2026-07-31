"""Cache the training runs from Weights & Biases into raw/wandb_runs.json.

The curve figures read the cache, never the network, so they redraw offline and a figure in the
post cannot silently change when a run is renamed or deleted upstream. Re-run this to refresh.

    uv run --with wandb python fetch_runs.py

Credentials come from ~/.netrc, the same place the training boxes log in with.
"""
from __future__ import annotations

import json
import pathlib

PROJECT = "binh-pham/sim2real-so-frame"
OUT = pathlib.Path(__file__).resolve().parent / "raw" / "wandb_runs.json"

# The runs the post cites. v4/v5 are the current recipe (200-step horizon, both jaw ramps in the
# reward); v1-v3 are the same code before it, kept because "the CNN never placed" is a claim the
# post makes and the flat curves are the evidence.
RUNS = (
    "v4-dino", "v4-squint", "v5-dino-global-mean", "v5-dino-global-cls",
    "v1-dino", "v1-squint", "v2-squint-s1", "v2-squint-s2", "v2-squint-s3",
    "v2-dino", "v3-squint",
)

CONFIG_KEYS = ("encoder", "num_envs", "num_updates", "batch_size", "res", "object_colors",
               "seed", "dino_pool", "encoder_lr", "total_timesteps", "replay_episodes")


def main() -> None:
    import wandb

    api = wandb.Api()
    out = {}
    for run in api.runs(PROJECT):
        if run.name not in RUNS:
            continue
        evals, train = [], []
        # scan_history without a key filter: the eval metrics are logged on their own steps, and
        # asking for them by name drops every row where they are absent.
        for row in run.scan_history(page_size=2000):
            step = row.get("_step")
            if step is None:
                continue
            if row.get("eval/success_at_end") is not None:
                evals.append({
                    "step": step,
                    "success_at_end": row.get("eval/success_at_end"),
                    "success_once": row.get("eval/success_once"),
                    "return": row.get("eval/return"),
                })
            if row.get("train/return") is not None:
                train.append({
                    "step": step,
                    "success_at_end": row.get("train/success_at_end"),
                    "return": row.get("train/return"),
                    "alpha": row.get("alpha"),
                    "q_max": row.get("q_max"),
                })
        out[run.name] = {
            "config": {k: run.config.get(k) for k in CONFIG_KEYS},
            "state": run.state,
            "created": str(run.created_at),
            "runtime_h": round(run.summary.get("_runtime", 0) / 3600, 2),
            "sps": run.summary.get("time/sps"),
            "eval": evals,
            "train": train,
        }
        best = max((e["success_at_end"] for e in evals), default=0.0)
        print(f"{run.name:22s} {len(evals):3d} evals  best {best:.3f}")

    missing = set(RUNS) - set(out)
    if missing:
        print(f"[warn] not found in {PROJECT}: {sorted(missing)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out) + "\n")
    print(f"[viz] wrote {OUT}")


if __name__ == "__main__":
    main()
