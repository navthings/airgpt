import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="plot loss and speed from a run's log")
    ap.add_argument("run", nargs="?", default="runs/hour")
    run = Path(ap.parse_args().run)
    rows = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]

    # sft minutes continue on from pretraining so both phases share one time axis
    offset = max((r["min"] for r in rows if r["phase"] == "pretrain"), default=0)
    t = lambda r: r["min"] + (offset if r["phase"] == "sft" else 0)
    train = [r for r in rows if "loss" in r]
    val = [r for r in rows if "val_loss" in r]

    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for phase, color in (("pretrain", "#3b6ea5"), ("sft", "#c4572e")):
        tr = [r for r in train if r["phase"] == phase]
        va = [r for r in val if r["phase"] == phase]
        a.plot([t(r) for r in tr], [r["loss"] for r in tr], color=color, alpha=0.5, label=f"{phase} train")
        a.plot([t(r) for r in va], [r["val_loss"] for r in va], "o-", color=color, label=f"{phase} val")
    a.set_ylabel("loss")
    a.legend()

    b.plot([t(r) for r in train], [r["tok_s"] / 1000 for r in train], color="#444")
    hot = [r for r in train if (r.get("thermal") or 0) >= 2]
    if hot:
        b.scatter([t(r) for r in hot], [r["tok_s"] / 1000 for r in hot], color="#d62728", s=12, label="thermal serious+")
        b.legend()
    b.set_ylabel("k tokens / s")
    b.set_xlabel("minutes")
    if offset:
        for ax in (a, b):
            ax.axvline(offset, color="#999", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(run / "plot.png", dpi=150)
    print(f"saved {run / 'plot.png'}")


if __name__ == "__main__":
    main()
