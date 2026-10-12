from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    vocab: int = 8192
    d: int = 256
    layers: int = 8
    heads: int = 4
    kv_heads: int = 2
    ctx: int = 512
    batch: int = 32
    accum: int = 1
    lr: float = 1e-3
    minutes: float = 60
    sft_frac: float = 0.2
    pretrain_tokens: int = 60_000_000
    sft_convos: int = 40_000
    mix: dict = field(default_factory=lambda: {"tinystories": 0.8, "cosmopedia": 0.2})

    @property
    def d_ff(self) -> int:
        # swiglu hidden size, ~8/3 of d rounded up to a multiple of 64
        return -(-int(8 * self.d / 3) // 64) * 64

    @property
    def gb_per_row(self) -> float:
        # rough peak training memory per batch row, fit to real runs on an m5 air (hour 5.8gb, night 5.4gb)
        return self.ctx * self.layers * (self.d * 131 + self.heads * self.ctx * 5.3) / 1e9

    def fit(self, ram_gb: float) -> "Config":
        # leave over half the ram for macos and everything else, split the batch until it fits
        self.accum = 1
        while self.batch // self.accum * self.gb_per_row > ram_gb * 0.45 and self.accum < self.batch:
            self.accum *= 2
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def parse_mix(text: str) -> dict:
    """'tinystories=0.7,wikipedia=0.3' or just 'fineweb-edu'. weights don't have to add up to 1."""
    mix = {}
    for part in text.split(","):
        name, _, w = part.strip().partition("=")
        mix[name] = float(w) if w else 1.0
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


PRESETS = {
    "smoke": Config(vocab=2048, d=128, layers=2, heads=4, kv_heads=2, ctx=128, batch=16, lr=2e-3,
                    minutes=3, pretrain_tokens=2_000_000, sft_convos=2_000),
    "hour": Config(),
    "night": Config(d=512, layers=12, heads=8, kv_heads=4, ctx=1024, batch=4, lr=6e-4,
                    minutes=480, pretrain_tokens=150_000_000, sft_convos=100_000,
                    mix={"tinystories": 0.5, "cosmopedia": 0.5}),
}
