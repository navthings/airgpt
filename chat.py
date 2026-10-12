import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

import ui
from config import Config
from model import GPT, count_params
from tokenizer import Tokenizer


def sample(logits: mx.array, temp: float, min_p: float) -> int:
    if temp == 0:
        return logits.argmax().item()
    probs = mx.softmax(logits.astype(mx.float32))
    # min-p before temperature (llama.cpp order): drop tokens under min_p x the top token's chance
    if min_p > 0:
        probs = mx.where(probs >= min_p * probs.max(), probs, 0)
    logp = mx.log(probs) / temp
    return mx.random.categorical(logp).item()


def generate(model, tok: Tokenizer, ids: list[int], ctx: int, max_new: int, temp: float, min_p: float):
    ids = ids[-(ctx - max_new):]
    logits, cache = model.generate_step(ids, None)
    out = []
    for _ in range(max_new):
        t = sample(logits, temp, min_p)
        if t == tok.end:
            break
        out.append(t)
        yield out
        logits, cache = model.generate_step([t], cache)


def stream(model, tok, ids, ctx, args) -> list[int]:
    shown, out = "", []
    for out in generate(model, tok, ids, ctx, args.max, args.temp, args.min_p):
        text = tok.decode(out)
        if not text.endswith("�"):  # wait for the rest of a split character
            sys.stdout.write(text[len(shown):])
            sys.stdout.flush()
            shown = text
    print()
    return out


def main():
    ap = argparse.ArgumentParser(description="talk to a trained airgpt model")
    ap.add_argument("run", nargs="?", default="runs/hour")
    ap.add_argument("--base", action="store_true", help="use the model from before chat training (continues text)")
    ap.add_argument("--prompt", help="answer one message and exit")
    # small models need colder sampling than big ones, see the liltemp paper
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--min-p", dest="min_p", type=float, default=0.1)
    ap.add_argument("--max", type=int, default=200, help="max new tokens per reply")
    args = ap.parse_args()

    run = Path(args.run)
    if not (run / "config.json").exists():
        raise SystemExit(f"no run in {run}. train one first: python train.py --preset smoke")
    cfg = Config(**json.loads((run / "config.json").read_text()))
    tok = Tokenizer.load(str(run / "tokenizer.json"))
    model = GPT(cfg)
    model.load_weights(str(run / ("base.safetensors" if args.base else "model.safetensors")))
    model.eval()

    if not args.prompt:
        try:
            import readline  # noqa: F401  arrow keys and history in input()
        except ImportError:
            pass
        n = count_params(model)
        print(f"\n{ui.bold('airgpt')} {run} ({n / 1e6:.2f}M params, temp {args.temp}, min-p {args.min_p})")
        print(ui.dim("base model, it continues whatever you type" if args.base else "/reset clears the chat, ctrl-c quits") + "\n")

    you = ui.readline_safe(ui.dim("> "))

    if args.base:
        while True:
            try:
                text = args.prompt or input(you)
            except (KeyboardInterrupt, EOFError):
                print()
                return
            sys.stdout.write(ui.dim(text))
            stream(model, tok, [tok.end] + tok.encode(text), cfg.ctx, args)
            if args.prompt:
                return
            print()

    history = [tok.end]
    while True:
        try:
            msg = args.prompt or input(you)
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if msg.strip() == "/reset":
            history = [tok.end]
            print(ui.dim("cleared\n"))
            continue
        if not msg.strip():
            continue
        history += [tok.user] + tok.encode(msg) + [tok.assistant]
        reply = stream(model, tok, history, cfg.ctx, args)
        history += reply + [tok.end]
        if args.prompt:
            return
        print()


if __name__ == "__main__":
    main()
