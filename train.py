import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import mac
from config import PRESETS, Config
from model import GPT, count_params


def lr_at(frac: float, peak: float, warm: float = 0.03) -> float:
    # linear warmup, then cosine down to 10% of peak, all driven by the clock not step counts
    if frac < warm:
        return peak * frac / warm
    p = min((frac - warm) / (1 - warm), 1.0)
    return peak * (0.1 + 0.45 * (1 + math.cos(math.pi * p)))


def loss_fn(model, x, y, m):
    ce = nn.losses.cross_entropy(model(x), y, reduction="none")
    return (ce * m).sum() / mx.maximum(m.sum(), 1)


class Data:
    def __init__(self, path: Path, cfg: Config, seed: int = 0):
        self.cfg, self.rng = cfg, np.random.default_rng(seed)
        self.meta = json.loads((path / "meta.json").read_text())
        self.train = np.memmap(path / "train.bin", dtype=np.uint16, mode="r")
        self.val = np.memmap(path / "val.bin", dtype=np.uint16, mode="r")
        self.chat = (path / "sft_x.npy").exists()
        if self.chat:
            self.sft_x, self.sft_m = np.load(path / "sft_x.npy"), np.load(path / "sft_m.npy")
            self.sft_vx, self.sft_vm = np.load(path / "sft_val_x.npy"), np.load(path / "sft_val_m.npy")

    def _windows(self, arr, starts):
        rows = np.stack([arr[s : s + self.cfg.ctx + 1] for s in starts]).astype(np.int32)
        return mx.array(rows[:, :-1]), mx.array(rows[:, 1:]), mx.ones(rows[:, 1:].shape)

    def pretrain(self):
        return self._windows(self.train, self.rng.integers(0, len(self.train) - self.cfg.ctx - 1, self.cfg.batch))

    def sft(self):
        i = self.rng.integers(0, len(self.sft_x), self.cfg.batch)
        x, m = self.sft_x[i].astype(np.int32), self.sft_m[i]
        return mx.array(x[:, :-1]), mx.array(x[:, 1:]), mx.array(m[:, 1:].astype(np.float32))

    def val_batches(self, phase: str, n: int = 16):
        if phase == "sft":
            for i in range(0, min(len(self.sft_vx), n * self.cfg.batch), self.cfg.batch):
                x, m = self.sft_vx[i : i + self.cfg.batch].astype(np.int32), self.sft_vm[i : i + self.cfg.batch]
                yield mx.array(x[:, :-1]), mx.array(x[:, 1:]), mx.array(m[:, 1:].astype(np.float32))
        else:
            starts = np.linspace(0, len(self.val) - self.cfg.ctx - 1, n * self.cfg.batch).astype(int)
            for j in range(n):
                yield self._windows(self.val, starts[j * self.cfg.batch : (j + 1) * self.cfg.batch])


def evaluate(model, data: Data, phase: str) -> float:
    total, count = 0.0, 0.0
    for x, y, m in data.val_batches(phase):
        ce = nn.losses.cross_entropy(model(x), y, reduction="none")
        total += (ce * m).sum().item()
        count += m.sum().item()
    return total / count


def save(out: Path, model, opt, progress: dict):
    model.save_weights(str(out / "model.safetensors"))
    mx.save_safetensors(str(out / "optim.safetensors"), dict(tree_flatten(opt.state)))
    (out / "progress.json").write_text(json.dumps(progress))


def log_line(out: Path, row: dict):
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def run_phase(phase: str, minutes: float, peak: float, model, data: Data, out: Path, progress: dict, warm: float):
    opt = optim.AdamW(learning_rate=0.0, betas=[0.9, 0.95], weight_decay=0.1)
    opt.init(model.trainable_parameters())
    if progress.get("phase") == phase and (out / "optim.safetensors").exists():
        opt.state = tree_unflatten(list(mx.load(str(out / "optim.safetensors")).items()))
        print(f"resuming {phase} at {progress['elapsed'] / 60:.1f} min")
    else:
        progress.update(phase=phase, elapsed=0.0, step=0, tokens=0)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, m):
        loss, grads = loss_and_grad(model, x, y, m)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        return loss

    next_batch = data.sft if phase == "sft" else data.pretrain
    total = minutes * 60
    start = time.time() - progress["elapsed"]
    last_log = last_eval = last_ckpt = time.time()
    log_tokens, log_t = 0, time.time()
    speeds, baseline, losses = [], None, []
    eval_every = max(60.0, total / 20)
    bpt = data.meta["bytes_per_token"]

    while (elapsed := time.time() - start) < total:
        opt.learning_rate = lr_at(elapsed / total, peak, warm)
        x, y, m = next_batch()
        loss = step(x, y, m)
        mx.eval(state, loss)
        losses.append(loss.item())
        progress["step"] += 1
        progress["tokens"] += x.size
        log_tokens += x.size

        now = time.time()
        if now - last_log >= 10:
            speed = log_tokens / (now - log_t)
            speeds.append((elapsed, speed))
            # baseline speed = the first couple of minutes, before a fanless chassis heats up
            if baseline is None and elapsed > 120:
                baseline = float(np.median([s for t, s in speeds if t > 20]))
            th = mac.thermal()
            vs = f" ({speed / baseline - 1:+.0%} vs start)" if baseline else ""
            row = {"phase": phase, "min": round(elapsed / 60, 3), "step": progress["step"],
                   "loss": round(float(np.mean(losses)), 4), "lr": opt.learning_rate.item(),
                   "tok_s": round(speed), "thermal": th, "mem_gb": round(mx.get_peak_memory() / 1e9, 2)}
            log_line(out, row)
            print(f"{phase:8} {elapsed / 60:6.1f}/{minutes:.1f} min | step {progress['step']:6} | "
                  f"loss {row['loss']:.3f} | {speed / 1000:5.1f}k tok/s{vs} | "
                  f"{mac.THERMAL.get(th, '-')} | {row['mem_gb']} GB")
            losses, log_tokens, log_t, last_log = [], 0, now, now
            if th == 3:
                print("thermal state critical: saving and cooling off for 30s")
                progress["elapsed"] = elapsed
                save(out, model, opt, progress)
                time.sleep(30)

        if now - last_eval >= eval_every:
            v = evaluate(model, data, phase)
            row = {"phase": phase, "min": round(elapsed / 60, 3), "val_loss": round(v, 4)}
            if phase == "pretrain":
                row["val_bpb"] = round(v / math.log(2) / bpt, 4)
            log_line(out, row)
            print(f"{'':8} val loss {v:.3f}" + (f" ({row['val_bpb']:.3f} bits per byte)" if "val_bpb" in row else ""))
            last_eval = time.time()

        if now - last_ckpt >= 600:
            progress["elapsed"] = time.time() - start
            save(out, model, opt, progress)
            last_ckpt = time.time()

    progress["elapsed"] = total
    final = evaluate(model, data, phase)
    log_line(out, {"phase": phase, "min": round(minutes, 3), "val_loss": round(final, 4)})
    save(out, model, opt, progress)
    return {"val_loss": round(final, 4), "steps": progress["step"], "tokens": progress["tokens"],
            "tok_s_start": round(baseline) if baseline else None,
            "tok_s_end": round(speeds[-1][1]) if speeds else None}


def main():
    ap = argparse.ArgumentParser(description="train a chatbot from scratch, for a fixed amount of time")
    ap.add_argument("--preset", default="hour", choices=PRESETS)
    ap.add_argument("--minutes", type=float, help="override the preset's time budget")
    ap.add_argument("--data", help="data folder (default data/<preset>, made automatically if missing)")
    ap.add_argument("--text", help="pretrain on your own .txt/.md file or folder")
    ap.add_argument("--chat", help="your own chat .jsonl")
    ap.add_argument("--out", help="run folder (default runs/<preset>)")
    ap.add_argument("--resume", action="store_true", help="continue the run in --out")
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    if a.minutes:
        cfg.minutes = a.minutes
    out = Path(a.out or f"runs/{a.preset}")
    data_dir = Path(a.data or f"data/{a.preset}" + (f"-{Path(a.text).stem}" if a.text else ""))

    progress = {}
    if (out / "progress.json").exists():
        if not a.resume:
            raise SystemExit(f"{out} already has a run. add --resume to continue it, or --out to start a new one")
        progress = json.loads((out / "progress.json").read_text())
        cfg = Config(**json.loads((out / "config.json").read_text()))
        if progress.get("phase") == "done":
            raise SystemExit(f"{out} is finished. chat with it: python chat.py {out}")

    if not (data_dir / "meta.json").exists():
        from prepare import prepare
        prepare(cfg, data_dir, a.text, a.chat)
    data = Data(data_dir, cfg)
    if data.meta["vocab"] != cfg.vocab or data.meta["ctx"] != cfg.ctx:
        raise SystemExit(f"{data_dir} was made for a different preset, point --data somewhere else")

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out / "tokenizer.json").write_bytes((data_dir / "tokenizer.json").read_bytes())

    model = GPT(cfg)
    if progress:
        model.load_weights(str(out / "model.safetensors"))
    hw = mac.info()
    n = count_params(model)
    print(f"\n{hw['model']} | {hw['chip']} | {hw['ram_gb']} GB" + (" | fanless, expect throttling on long runs" if hw["fanless"] else ""))
    print(f"model: {n / 1e6:.1f}M params, {cfg.layers} layers, d {cfg.d}, context {cfg.ctx}")
    print(f"budget: {cfg.minutes:.0f} min ({'pretrain then chat' if data.chat else 'pretrain only'})\n")

    pre_min = cfg.minutes * (1 - cfg.sft_frac) if data.chat else cfg.minutes
    summary = {"hardware": hw, "params": n, "preset": a.preset, "minutes": cfg.minutes}
    t0 = time.time()
    if progress.get("phase") != "sft":
        summary["pretrain"] = run_phase("pretrain", pre_min, cfg.lr, model, data, out, progress, warm=0.03)
        model.save_weights(str(out / "base.safetensors"))
    if data.chat:
        summary["chat"] = run_phase("sft", cfg.minutes - pre_min, cfg.lr * 0.3, model, data, out, progress, warm=0.05)

    progress["phase"] = "done"
    (out / "progress.json").write_text(json.dumps(progress))
    summary["peak_mem_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    summary["wall_minutes"] = round((time.time() - t0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone in {summary['wall_minutes']} min, peak memory {summary['peak_mem_gb']} GB")
    print(f"chat:  python chat.py {out}")
    print(f"plot:  python plot.py {out}")


if __name__ == "__main__":
    main()
