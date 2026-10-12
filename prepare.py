import argparse
import json
import random
import time
from itertools import chain
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import ui
from config import PRESETS, Config
from tokenizer import SPECIALS, Tokenizer

# name, config, split, field on the Hugging Face hub
SOURCES = {
    "tinystories": ("roneneldan/TinyStories", None, "train", "text"),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train", "text"),
    "everyday": ("HuggingFaceTB/everyday-conversations-llama3.1-2k", None, "train_sft", "messages"),
    "smoltalk": ("HuggingFaceTB/smol-smoltalk", None, "train", "messages"),
}


def hf_stream(key: str):
    from datasets import load_dataset
    name, config, split, field = SOURCES[key]
    for ex in load_dataset(name, config, split=split, streaming=True):
        yield ex[field]


def mixed(mix: dict):
    # always pull from whichever source is furthest behind its share, so the mix holds by amount of text, not doc count
    streams = {k: hf_stream(k) for k in mix}
    chars = {k: 0 for k in mix}
    while streams:
        k = min(streams, key=lambda s: chars[s] / mix[s])
        try:
            doc = next(streams[k])
        except StopIteration:
            del streams[k]
            continue
        chars[k] += len(doc)
        yield doc


def local_text(path: str):
    p = Path(path)
    files = [p] if p.is_file() else sorted(chain(p.rglob("*.txt"), p.rglob("*.md")))
    if not files:
        raise SystemExit(f"no .txt or .md files found in {path}")
    for f in files:
        # cut long files into ~4k character pieces on paragraph breaks
        buf = ""
        for para in f.read_text(errors="replace").split("\n\n"):
            buf += para + "\n\n"
            if len(buf) > 4000:
                yield buf.strip()
                buf = ""
        if buf.strip():
            yield buf.strip()


def local_chat(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)["messages"]


_tok: Tokenizer | None = None


def _init(path: str):
    global _tok
    _tok = Tokenizer.load(path)


def _enc_doc(text: str) -> list[int]:
    return _tok.encode(text) + [_tok.end]


def _enc_chat(messages: list[dict]) -> tuple[list[int], list[int]]:
    # <|end|> <|user|> hi <|assistant|> hello! <|end|> ... loss only on the assistant's words
    ids, mask = [_tok.end], [0]
    for m in messages:
        if m["role"] == "user":
            t = [_tok.user] + _tok.encode(m["content"])
            ids += t
            mask += [0] * len(t)
        elif m["role"] == "assistant":
            t = _tok.encode(m["content"]) + [_tok.end]
            ids += [_tok.assistant] + t
            mask += [0] + [1] * len(t)
    return ids, mask


def take_chars(docs, n_chars: int, on_progress=None) -> list[str]:
    out, total = [], 0
    for d in docs:
        out.append(d)
        total += len(d)
        if on_progress:
            on_progress(len(d))
        if total >= n_chars:
            break
    return out


def take_n(it, n: int, on_progress=None) -> list:
    out = []
    for x in it:
        out.append(x)
        if on_progress:
            on_progress(len(out))
        if len(out) >= n:
            break
    return out


def batched(it, n: int):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def pack(convos, ctx: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    # greedily fit whole conversations into rows of ctx+1 tokens, pad the rest (masked out)
    rows, masks, cur, cur_m = [], [], [], []
    for ids, m in convos:
        ids, m = ids[: ctx + 1], m[: ctx + 1]
        if sum(m) == 0:
            continue
        if len(cur) + len(ids) > ctx + 1:
            rows.append(cur + [end] * (ctx + 1 - len(cur)))
            masks.append(cur_m + [0] * (ctx + 1 - len(cur_m)))
            cur, cur_m = [], []
        cur, cur_m = cur + ids, cur_m + m
    if cur:
        rows.append(cur + [end] * (ctx + 1 - len(cur)))
        masks.append(cur_m + [0] * (ctx + 1 - len(cur_m)))
    return np.array(rows, dtype=np.uint16), np.array(masks, dtype=np.uint8)


def prepare(cfg: Config, out: Path, text: str | None = None, chat: str | None = None, seed: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    live = ui.Live()
    print(f"\n{ui.bold('making')} {out}")
    if text:
        docs = local_text(text)
        print(f"source: {text}")
    else:
        docs = mixed(cfg.mix)
        print("source: " + ", ".join(f"{k} {v:.0%}" for k, v in cfg.mix.items()) + " (hugging face)")

    val_tokens = max(cfg.ctx * 64, cfg.pretrain_tokens // 200)
    want = val_tokens * 4 + min(20_000_000, cfg.pretrain_tokens * 2)
    got = [0]

    def downloaded(n):
        got[0] += n
        if live.due():
            live.show(f"download   {ui.bar(got[0] / want, 30)} {ui.num(got[0])}/{ui.num(want)} chars")

    val_docs = take_chars(docs, val_tokens * 4, downloaded)
    tok_docs = take_chars(docs, want - val_tokens * 4, downloaded)
    live.done(f"downloaded {ui.num(got[0])} chars")

    t = time.time()
    live.show("tokenizer  counting words")
    tok = Tokenizer.train(tok_docs, cfg.vocab, lambda done, total: live.due() and live.show(
        f"tokenizer  {ui.bar(done / total, 30)} {done}/{total} merges"))
    tok.save(str(out / "tokenizer.json"))
    live.done(f"tokenizer: {len(tok.merges) + 256 + len(SPECIALS)} tokens in {time.time() - t:.0f}s")

    with Pool(initializer=_init, initargs=(str(out / "tokenizer.json"),)) as pool:
        val = np.array([i for ids in pool.imap(_enc_doc, val_docs, chunksize=16) for i in ids], dtype=np.uint16)
        val.tofile(out / "val.bin")
        bytes_per_token = sum(len(d.encode()) + 1 for d in val_docs) / len(val)
        print(f"val: {ui.num(len(val))} tokens, {bytes_per_token:.2f} bytes/token")

        n, t = 0, time.time()
        with open(out / "train.bin", "wb") as f:
            # feed the pool in batches so a fast download can't pile up the whole dataset in RAM
            for batch in batched(chain(tok_docs, docs), 4096):
                for ids in pool.imap(_enc_doc, batch, chunksize=64):
                    np.array(ids, dtype=np.uint16).tofile(f)
                    n += len(ids)
                rate = n / max(time.time() - t, 1e-9)
                live.show(f"tokenizing {ui.bar(n / cfg.pretrain_tokens, 30)} {ui.num(n)}/{ui.num(cfg.pretrain_tokens)}"
                          f" tokens, {ui.num(rate)}/s")
                if n >= cfg.pretrain_tokens:
                    break
        live.done(f"train: {ui.num(n)} tokens in {ui.clock(time.time() - t)}")

        if chat or not text:
            if chat:
                convos = list(local_chat(chat))
            else:
                everyday = list(hf_stream("everyday"))
                # tiny models can only learn short chats, so keep the short ones
                short = (m for m in hf_stream("smoltalk") if sum(len(x["content"]) for x in m) < cfg.ctx * 3)
                found = take_n(short, cfg.sft_convos, lambda k: live.due() and live.show(
                    f"chat       {ui.bar(k / cfg.sft_convos, 30)} {k}/{cfg.sft_convos} conversations"))
                convos = everyday * 2 + found
            random.Random(seed).shuffle(convos)
            encoded = pool.map(_enc_chat, convos, chunksize=64)
            x, m = pack(encoded, cfg.ctx, tok.end)
            n_val = max(8, len(x) // 50)
            np.save(out / "sft_x.npy", x[n_val:])
            np.save(out / "sft_m.npy", m[n_val:])
            np.save(out / "sft_val_x.npy", x[:n_val])
            np.save(out / "sft_val_m.npy", m[:n_val])
            live.done(f"chat: {len(convos)} conversations -> {len(x)} rows")

    meta = {"vocab": cfg.vocab, "ctx": cfg.ctx, "train_tokens": n, "val_tokens": len(val),
            "bytes_per_token": bytes_per_token, "chat": bool(chat or not text),
            "source": text or cfg.mix, "minutes_to_prepare": round((time.time() - t0) / 60, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"data ready in {ui.clock(time.time() - t0)}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="download, tokenize and pack training data")
    ap.add_argument("--preset", default="hour", choices=PRESETS)
    ap.add_argument("--text", help="train on your own .txt/.md file or folder instead of the default data")
    ap.add_argument("--chat", help="your own chat data: .jsonl with {'messages': [{'role', 'content'}]} per line")
    ap.add_argument("--out", help="where to write (default data/<preset>)")
    a = ap.parse_args()
    prepare(PRESETS[a.preset], Path(a.out or f"data/{a.preset}"), a.text, a.chat)
