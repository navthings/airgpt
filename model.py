import mlx.core as mx
import mlx.nn as nn

from config import Config


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.h, self.kv = cfg.heads, cfg.kv_heads
        self.hd = cfg.d // cfg.heads
        self.wq = nn.Linear(cfg.d, cfg.d, bias=False)
        self.wk = nn.Linear(cfg.d, self.kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d, self.kv * self.hd, bias=False)
        self.wo = nn.Linear(cfg.d, cfg.d, bias=False)
        self.rope = nn.RoPE(self.hd, base=10000)

    def __call__(self, x, cache=None):
        B, L, _ = x.shape
        q = self.wq(x).reshape(B, L, self.h, self.hd).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, L, self.kv, self.hd).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, L, self.kv, self.hd).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache[0].shape[2]
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        # grouped-query attention: fewer k/v heads than q heads, sdpa handles the sharing
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.hd**-0.5, mask="causal" if L > 1 else None)
        return self.wo(o.transpose(0, 2, 1, 3).reshape(B, L, -1)), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d, eps=1e-5)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d, eps=1e-5)
        self.mlp = MLP(cfg)

    def __call__(self, x, cache=None):
        h, cache = self.attn(self.attn_norm(x), cache)
        x = x + h
        return x + self.mlp(self.mlp_norm(x)), cache


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = [Block(cfg) for _ in range(cfg.layers)]
        self.norm = nn.RMSNorm(cfg.d, eps=1e-5)

    def __call__(self, x, caches: list | None = None):
        h = self.embed(x)
        new = []
        for i, block in enumerate(self.blocks):
            h, c = block(h, None if caches is None else caches[i])
            new.append(c)
        # tied embeddings: the input table doubles as the output layer
        logits = self.embed.as_linear(self.norm(h))
        return logits if caches is None else (logits, new)

    def generate_step(self, ids: list[int], caches: list | None):
        """Feed ids (whole prompt first, then one token at a time), get last logits + cache."""
        logits, new = self(mx.array([ids]), caches or [None] * len(self.blocks))
        return logits[0, -1], new


def count_params(model: nn.Module) -> int:
    from mlx.utils import tree_flatten
    return sum(v.size for _, v in tree_flatten(model.parameters()))
