import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, INK, GREY = "#2f6db5", "#d0643a", "#222222", "#9a9a9a"
HEAT = {1: "#e0a800", 2: "#d62728", 3: "#8b0000"}


def main():
    ap = argparse.ArgumentParser(description="plot loss and speed from a run's log")
    ap.add_argument("run", nargs="?", default="runs/hour")
    run = Path(ap.parse_args().run)
    rows = [json.loads(l) for l in (run / "log.jsonl").read_text().splitlines()]
    cfg = json.loads((run / "config.json").read_text())
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").exists() else {}

    # chat minutes carry on from pretraining so both phases share one time axis
    offset = max((r["min"] for r in rows if r["phase"] == "pretrain"), default=0)
    t = lambda r: r["min"] + (offset if r["phase"] == "sft" else 0)
    train = [r for r in rows if "loss" in r]
    val = [r for r in rows if "val_loss" in r]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})

    for phase, color, name in (("pretrain", BLUE, "pretraining"), ("sft", ORANGE, "chat")):
        tr = [r for r in train if r["phase"] == phase]
        va = [r for r in val if r["phase"] == phase]
        if not tr:
            continue
        a.plot([t(r) for r in tr], [r["loss"] for r in tr], color=color, alpha=0.35, lw=1)
        a.plot([t(r) for r in va], [r["val_loss"] for r in va], "o-", color=color, ms=3.5, lw=1.6, label=name)
        if va:
            a.annotate(f"{va[-1]['val_loss']:.2f}", (t(va[-1]), va[-1]["val_loss"]), xytext=(6, 0),
                       textcoords="offset points", va="center", color=color, fontsize=9)
    a.set_ylabel("loss")
    a.legend(frameon=False, loc="upper right")
    a.grid(axis="y", color="#ececec")

    b.plot([t(r) for r in train], [r["tok_s"] / 1000 for r in train], color=INK, lw=1.2)
    for level, color in HEAT.items():
        hot = [r for r in train if r.get("thermal") == level]
        if hot:
            b.scatter([t(r) for r in hot], [r["tok_s"] / 1000 for r in hot], color=color, s=10, zorder=3,
                      label=f"thermal: {['', 'fair', 'serious', 'critical'][level]}")
    if any(r.get("thermal") for r in train):
        b.legend(frameon=False, loc="lower left", fontsize=9)
    b.set_ylabel("thousand tokens / s")
    b.set_xlabel("minutes")
    b.set_ylim(bottom=0)
    b.grid(axis="y", color="#ececec")

    if offset and any(r["phase"] == "sft" for r in rows):
        for ax in (a, b):
            ax.axvspan(offset, max(t(r) for r in rows), color=ORANGE, alpha=0.05, lw=0)

    hw = summary.get("hardware", {})
    params = summary.get("params")
    sub = ", ".join(x for x in [hw.get("model"), hw.get("chip"), f"{hw['ram_gb']} GB" if hw else None,
                                 f"{params / 1e6:.1f}M params" if params else None,
                                 f"{cfg['minutes']:.0f} min"] if x)
    fig.suptitle("airgpt", x=0.065, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.065, 0.935, sub, ha="left", fontsize=10, color=GREY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(run / "plot.png", dpi=200, facecolor="white")
    print(f"saved {run / 'plot.png'}")


if __name__ == "__main__":
    main()
