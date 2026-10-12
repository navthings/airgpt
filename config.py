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

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "smoke": Config(vocab=2048, d=128, layers=2, heads=4, kv_heads=2, ctx=128, batch=16, lr=2e-3,
                    minutes=3, pretrain_tokens=2_000_000, sft_convos=2_000),
    "hour": Config(),
    "night": Config(d=512, layers=12, heads=8, kv_heads=4, ctx=1024, batch=16, lr=6e-4,
                    minutes=480, pretrain_tokens=400_000_000, sft_convos=150_000,
                    mix={"tinystories": 0.5, "cosmopedia": 0.5}),
}
