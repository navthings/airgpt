import argparse
import time

import mlx.core as mx
import mlx.optimizers as optim

import mac
import ui
from config import PRESETS
from model import GPT, count_params
from train import make_step


def main():
    ap = argparse.ArgumentParser(description="how fast does this mac train? 2 minutes, no download")
    ap.add_argument("--preset", default="hour", choices=PRESETS)
    ap.add_argument("--seconds", type=float, default=120)
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    hw = mac.info()
    cfg.fit(hw["ram_gb"])
    model = GPT(cfg)
    opt = optim.AdamW(learning_rate=cfg.lr, betas=[0.9, 0.95], weight_decay=0.1)
    opt.init(model.trainable_parameters())
    step, state = make_step(model, opt, cfg.accum)

    # random tokens, the speed doesn't care what the text says
    rows = mx.random.randint(0, cfg.vocab, (cfg.batch, cfg.ctx + 1))
    x, y, m = rows[:, :-1], rows[:, 1:], mx.ones((cfg.batch, cfg.ctx))
    for _ in range(5):
        mx.eval(state, step(x, y, m))

    mac.keep_awake()
    name = f"{hw['model']} ({hw['chip']}, {hw['ram_gb']}GB)"
    split = f", batch in {cfg.accum} pieces to fit {hw['ram_gb']}GB" if cfg.accum > 1 else ""
    print(f"\n{ui.bold('airgpt bench')} {a.preset}, {count_params(model) / 1e6:.1f}M params, {a.seconds:g}s{split}")
    print(name + "\n")

    live, windows = ui.Live(), []
    start = w_start = time.time()
    w_tokens, th = 0, mac.thermal()
    while (elapsed := time.time() - start) < a.seconds:
        mx.eval(state, step(x, y, m))
        w_tokens += x.size
        now = time.time()
        if now - w_start >= 10:
            windows.append(w_tokens / (now - w_start))
            w_start, w_tokens, th = now, 0, mac.thermal()
        if live.due():
            speed = ui.num(windows[-1]) + " tok/s" if windows else "warming up"
            live.show(f"{ui.bar(elapsed / a.seconds, 30)} {ui.clock(elapsed)}/{ui.clock(a.seconds)}   {speed}   "
                      + ui.thermal(th))
    live.done()

    k = max(1, len(windows) // 4)
    first, last = sum(windows[:k]) / k, sum(windows[-k:]) / k
    drop = last / first - 1
    print(f"{ui.bold(ui.num(last))} tok/s, {drop:+.0%} from the start, thermal {ui.THERMAL.get(th, '-')}, mlx {mx.__version__}")
    print(ui.dim("\nfor the table in the readme:"))
    print(f"| {hw['model']} | {hw['chip']} | {hw['ram_gb']}GB | `{a.preset}` | {ui.num(last)} | {drop:+.0%} | "
          f"{ui.THERMAL.get(th, '-')} |\n")


if __name__ == "__main__":
    main()
