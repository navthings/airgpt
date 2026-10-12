import argparse
import json
import math
import shutil
import sys
import time
from collections import deque
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

import mac
import ui
from chat import generate
from config import PRESETS, Config
from model import GPT, count_params
from tokenizer import Tokenizer

CHAT_PROBE = "hi! what's your favourite animal?"


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


def preview(model, tok: Tokenizer, ids: list[int], ctx: int) -> str:
    # same seed every time, so the samples are comparable as training goes on
    mx.random.seed(0)
    out = []
    for out in generate(model, tok, ids, ctx, 48, 0.6, 0.1):
        pass
    return tok.decode(out)


def save(out: Path, model, opt, progress: dict):
    model.save_weights(str(out / "model.safetensors"))
    mx.save_safetensors(str(out / "optim.safetensors"), dict(tree_flatten(opt.state)))
    (out / "progress.json").write_text(json.dumps(progress))


def log_line(out: Path, row: dict):
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


TABLE = f"  {'time':>7}  {'step':>8}  {'loss':>7}  {'val':>7}  {'tok/s':>7}  {'':>5}  {'thermal':<8}  {'memory':>7}"


def run_phase(phase: str, minutes: float, peak: float, model, data: Data, tok: Tokenizer, out: Path,
              progress: dict, warm: float, prompt: list[int], prefix: str):
    opt = optim.AdamW(learning_rate=0.0, betas=[0.9, 0.95], weight_decay=0.1)
    opt.init(model.trainable_parameters())
    if progress.get("phase") == phase and (out / "optim.safetensors").exists():
        opt.state = tree_unflatten(list(mx.load(str(out / "optim.safetensors")).items()))
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

    name, what = ("pretrain", "reading text, learning to predict the next token") if phase == "pretrain" else \
                 ("chat", "learning to answer as the assistant")
    resumed = f", resuming at {ui.clock(progress['elapsed'])}" if progress["elapsed"] else ""
    print(f"\n{ui.bold(name)}  {ui.dim(f'{minutes:.0f} min · {what}{resumed}')}\n")
    if phase == "sft":
        print(f"  {ui.dim('every row asks it:')} {CHAT_PROBE}\n")
    print(ui.dim(TABLE))

    next_batch = data.sft if phase == "sft" else data.pretrain
    total = minutes * 60
    row_every = max(20.0, total / 24)
    start = time.time() - progress["elapsed"]
    next_row = progress["elapsed"] + row_every
    last_log = last_ckpt = time.time()
    recent = deque([(time.time(), progress["tokens"])], maxlen=64)
    speeds, baseline, losses, row_losses = [], None, [], []
    speed, th, live = 0.0, mac.thermal(), ui.Live()

    try:
        while (elapsed := time.time() - start) < total:
            opt.learning_rate = lr_at(elapsed / total, peak, warm)
            x, y, m = next_batch()
            loss = step(x, y, m)
            mx.eval(state, loss)
            losses.append(loss.item())
            row_losses.append(losses[-1])
            progress["step"] += 1
            progress["tokens"] += x.size

            now = time.time()
            recent.append((now, progress["tokens"]))
            while len(recent) > 2 and now - recent[0][0] > 10:
                recent.popleft()
            if now - recent[0][0] > 0.5:
                speed = (recent[-1][1] - recent[0][1]) / (now - recent[0][0])

            if now - last_log >= 10:
                th = mac.thermal()
                speeds.append((elapsed, speed))
                # speed over the first two minutes is the baseline, before a fanless chassis heats up
                if baseline is None and elapsed > 120:
                    baseline = float(np.median([s for t, s in speeds if t > 20]))
                log_line(out, {"phase": phase, "min": round(elapsed / 60, 3), "step": progress["step"],
                               "loss": round(float(np.mean(losses)), 4), "lr": opt.learning_rate.item(),
                               "tok_s": round(speed), "thermal": th, "mem_gb": round(mx.get_peak_memory() / 1e9, 2)})
                losses, last_log = [], now
                if th == 3:
                    live.say("  " + ui.red("thermal state critical, saving and cooling off for 30s"))
                    progress["elapsed"] = elapsed
                    save(out, model, opt, progress)
                    time.sleep(30)
                    start += 30

            if now - live.last > 0.1:
                live.show(ui.fit([f"  {ui.bar(elapsed / total)}", f"{ui.clock(elapsed)} / {ui.clock(total)}",
                                  f"loss {np.mean(row_losses[-50:]):.3f}", f"{ui.num(speed)} tok/s",
                                  ui.thermal(th), f"step {progress['step']:,}"]))

            if elapsed >= next_row:
                v = evaluate(model, data, phase)
                text = preview(model, tok, prompt, data.cfg.ctx)
                row = {"phase": phase, "min": round(elapsed / 60, 3), "val_loss": round(v, 4)}
                if phase == "pretrain":
                    row["val_bpb"] = round(v / math.log(2) / data.meta["bytes_per_token"], 4)
                log_line(out, {**row, "sample": text})
                vs = ""
                if baseline:
                    pct = speed / baseline - 1
                    vs = f"{pct:+.0%}".rjust(5)
                    vs = ui.red(vs) if pct < -0.25 else ui.yellow(vs) if pct < -0.1 else ui.dim(vs)
                else:
                    vs = " " * 5
                live.say(f"  {ui.clock(elapsed):>7}  {progress['step']:>8,}  {np.mean(row_losses):>7.3f}  "
                         f"{ui.bold(f'{v:>7.3f}')}  {ui.num(speed):>7}  {vs}  {ui.thermal(th, 8)}  "
                         f"{mx.get_peak_memory() / 1e9:>4.1f} GB")
                live.say(" " * 12 + ui.faint(ui.clip(prefix + text, ui.width() - 14)))
                row_losses = []
                next_row += row_every

            if now - last_ckpt >= 600:
                progress["elapsed"] = time.time() - start
                save(out, model, opt, progress)
                last_ckpt = time.time()
    except KeyboardInterrupt:
        progress["elapsed"] = time.time() - start
        save(out, model, opt, progress)
        args = [a for a in sys.argv[1:] if a not in ("--restart", "--resume")]
        live.done(f"\n  stopped at {ui.clock(progress['elapsed'])} and saved. carry on with\n\n"
                  f"  {ui.bold(' '.join(['python', 'train.py', *args, '--resume']))}\n")
        raise SystemExit(130)

    live.done()
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
    ap.add_argument("--restart", action="store_true", help="delete the run in --out and start over")
    a = ap.parse_args()

    cfg = PRESETS[a.preset]
    if a.minutes:
        cfg.minutes = a.minutes
    out = Path(a.out or f"runs/{a.preset}")
    data_dir = Path(a.data or f"data/{a.preset}" + (f"-{Path(a.text).stem}" if a.text else ""))

    if a.restart and out.exists():
        shutil.rmtree(out)
    progress = {}
    if (out / "progress.json").exists():
        if not a.resume:
            raise SystemExit(f"{out} already has a run. add --resume to continue it, or --restart to start over")
        progress = json.loads((out / "progress.json").read_text())
        cfg = Config(**json.loads((out / "config.json").read_text()))
        if progress.get("phase") == "done":
            raise SystemExit(f"{out} is finished. talk to it with: python chat.py {out}")

    print(f"\n{ui.bold('airgpt')}  {ui.dim(a.preset + ' preset')}")
    if not (data_dir / "meta.json").exists():
        from prepare import prepare
        prepare(cfg, data_dir, a.text, a.chat)
    data = Data(data_dir, cfg)
    if data.meta["vocab"] != cfg.vocab or data.meta["ctx"] != cfg.ctx:
        raise SystemExit(f"{data_dir} was made for a different preset, point --data somewhere else")

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (out / "tokenizer.json").write_bytes((data_dir / "tokenizer.json").read_bytes())
    tok = Tokenizer.load(str(out / "tokenizer.json"))

    model = GPT(cfg)
    if progress:
        model.load_weights(str(out / "model.safetensors"))
    hw = mac.info()
    awake = mac.keep_awake()
    n = count_params(model)
    pre_min = cfg.minutes * (1 - cfg.sft_frac) if data.chat else cfg.minutes
    chat_rows = f" · {len(data.sft_x):,} chat rows" if data.chat else ""
    print()
    print(ui.label("mac", f"{hw['model']} · {hw['chip']} · {hw['ram_gb']} GB"
                   + (ui.dim(" · fanless, so expect it to slow down as it warms up") if hw["fanless"] else "")))
    print(ui.label("model", f"{ui.num(n)} parameters · {cfg.layers} layers · width {cfg.d} · context {cfg.ctx}"))
    print(ui.label("data", f"{ui.num(data.meta['train_tokens'])} tokens{chat_rows} · {data_dir}"))
    print(ui.label("time", f"{cfg.minutes:.0f} min" + (f" · {pre_min:.0f} pretraining, {cfg.minutes - pre_min:.0f} chat"
                                                        if data.chat else "")
                   + (ui.dim(" · keeping the mac awake") if awake else "")))
    print(ui.label("saving to", str(out)))

    # what the samples under each row start from
    if isinstance(data.meta["source"], dict) and "tinystories" in data.meta["source"]:
        story = tok.encode("Once upon a time")
    else:
        story = [int(i) for i in data.val[:8]]
    probe = [tok.end, tok.user] + tok.encode(CHAT_PROBE) + [tok.assistant]

    summary = {"hardware": hw, "params": n, "preset": a.preset, "minutes": cfg.minutes}
    t0 = time.time()
    if progress.get("phase") != "sft":
        summary["pretrain"] = run_phase("pretrain", pre_min, cfg.lr, model, data, tok, out, progress, 0.03,
                                        [tok.end] + story, tok.decode(story))
        model.save_weights(str(out / "base.safetensors"))
    if data.chat:
        summary["chat"] = run_phase("sft", cfg.minutes - pre_min, cfg.lr * 0.3, model, data, tok, out, progress, 0.05,
                                    probe, "")

    progress["phase"] = "done"
    (out / "progress.json").write_text(json.dumps(progress))
    summary["peak_mem_gb"] = round(mx.get_peak_memory() / 1e9, 2)
    summary["wall_minutes"] = round((time.time() - t0) / 60, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    p = summary["pretrain"] if "pretrain" in summary else None
    print(f"\n{ui.green('done')}  {ui.dim(ui.clock(time.time() - t0))}\n")
    if p:
        bpb = p["val_loss"] / math.log(2) / data.meta["bytes_per_token"]
        speed = f"{ui.num(p['tok_s_start'])} -> {ui.num(p['tok_s_end'])} tok/s" if p["tok_s_start"] else ""
        print(ui.label("pretrain", f"val loss {p['val_loss']:.3f} · {bpb:.2f} bits per byte · "
                                   f"{ui.num(p['tokens'])} tokens" + (f" · {speed}" if speed else "")))
    if "chat" in summary:
        print(ui.label("chat", f"val loss {summary['chat']['val_loss']:.3f} · {ui.num(summary['chat']['tokens'])} tokens"))
    print(ui.label("memory", f"{summary['peak_mem_gb']} GB peak"))
    print()
    print(ui.label("talk to it", ui.bold(f"python chat.py {out}")))
    print(ui.label("plot it", ui.bold(f"python plot.py {out}")))
    print()


if __name__ == "__main__":
    main()
