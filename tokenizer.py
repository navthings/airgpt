import heapq
import json
import re
from collections import Counter, defaultdict

# split text into chunks so merges never cross words. space sticks to the front of a word,
# digits go one at a time (so numbers stay learnable), trailing spaces don't eat the next word's space
PATTERN = r" ?[^\W\d_]+| ?\d| ?(?:[^\w\s]|_)+|\s+(?!\S)|\s+"
SPECIALS = ["<|end|>", "<|user|>", "<|assistant|>"]


def merge(ids: list[int], pair: tuple[int, int], new: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges: dict[tuple[int, int], int], vocab_size: int):
        self.merges = merges
        self.vocab_size = vocab_size
        self.pat = re.compile(PATTERN)
        self.special = {s: vocab_size - len(SPECIALS) + i for i, s in enumerate(SPECIALS)}
        self.end, self.user, self.assistant = (self.special[s] for s in SPECIALS)
        self.bytes = {i: bytes([i]) for i in range(256)}
        for (a, b), new in sorted(merges.items(), key=lambda kv: kv[1]):
            self.bytes[new] = self.bytes[a] + self.bytes[b]
        for s, i in self.special.items():
            self.bytes[i] = s.encode()
        self.cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, texts: list[str], vocab_size: int, progress=None) -> "Tokenizer":
        pat = re.compile(PATTERN)
        counts = Counter(c for t in texts for c in pat.findall(t))
        words = [list(c.encode()) for c in counts]
        freqs = list(counts.values())

        # pair counts over unique chunks x how often each chunk appears
        pairs: Counter = Counter()
        where = defaultdict(set)
        for i, w in enumerate(words):
            for p in zip(w, w[1:]):
                pairs[p] += freqs[i]
                where[p].add(i)
        heap = [(-c, p) for p, c in pairs.items()]
        heapq.heapify(heap)

        merges = {}
        target = vocab_size - len(SPECIALS)
        for new in range(256, target):
            while heap:
                c, p = heapq.heappop(heap)
                if pairs.get(p, 0) == -c:  # skip stale heap entries
                    break
            else:
                break
            if -c < 2:
                break
            merges[p] = new
            touched = set()
            for i in where.pop(p):
                w, f = words[i], freqs[i]
                if p not in zip(w, w[1:]):
                    continue
                for q in zip(w, w[1:]):
                    pairs[q] -= f
                    touched.add(q)
                w = merge(w, p, new)
                words[i] = w
                for q in zip(w, w[1:]):
                    pairs[q] += f
                    where[q].add(i)
                    touched.add(q)
            pairs.pop(p, None)
            for q in touched:
                if pairs.get(q, 0) > 0:
                    heapq.heappush(heap, (-pairs[q], q))
            if progress:
                progress(new - 255, target - 256)
        return cls(merges, vocab_size)

    def _chunk(self, chunk: str) -> list[int]:
        ids = self.cache.get(chunk)
        if ids is not None:
            return ids
        ids = list(chunk.encode())
        while len(ids) > 1:
            # merges were learned in id order, so the lowest id is the earliest merge
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, 1 << 30))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        if len(self.cache) < 500_000:
            self.cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        return [i for c in self.pat.findall(text) for i in self._chunk(c)]

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size, "pattern": PATTERN, "specials": SPECIALS,
                       "merges": [[a, b, n] for (a, b), n in self.merges.items()]}, f)

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path) as f:
            d = json.load(f)
        return cls({(a, b): n for a, b, n in d["merges"]}, d["vocab_size"])
