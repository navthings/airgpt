import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx

from config import Config
from model import GPT
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
    ap.add_argument("--base", action="store_true", help="use the pretrained model before chat training (plain text continuation)")
    ap.add_argument("--prompt", help="answer one message and exit")
    # small models need colder sampling than big ones: see the liltemp paper
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--min-p", dest="min_p", type=float, default=0.1)
    ap.add_argument("--max", type=int, default=200, help="max new tokens per reply")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = Config(**json.loads((run / "config.json").read_text()))
    tok = Tokenizer.load(str(run / "tokenizer.json"))
    model = GPT(cfg)
    weights = run / ("base.safetensors" if args.base else "model.safetensors")
    model.load_weights(str(weights))
    model.eval()

    if args.base:
        prompt = args.prompt or input("start of text: ")
        print(prompt, end="")
        stream(model, tok, [tok.end] + tok.encode(prompt), cfg.ctx, args)
        return

    history = [tok.end]
    msgs = [args.prompt] if args.prompt else None
    print(f"chatting with {run} (temp {args.temp}, min-p {args.min_p}). ctrl-c to quit, /reset to clear\n")
    while True:
        try:
            msg = msgs.pop(0) if msgs is not None else input("you: ")
        except (KeyboardInterrupt, EOFError):
            print()
            return
        if msg.strip() == "/reset":
            history = [tok.end]
            continue
        history += [tok.user] + tok.encode(msg) + [tok.assistant]
        print("bot: ", end="", flush=True)
        reply = stream(model, tok, history, cfg.ctx, args)
        history += reply + [tok.end]
        if msgs is not None and not msgs:
            return


if __name__ == "__main__":
    main()
