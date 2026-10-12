import os
import re
import shutil
import sys
import time

TTY = sys.stdout.isatty()
COLOR = TTY and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"
ANSI = re.compile(r"\033\[[0-9;]*m")


def style(code: str):
    return lambda s: f"\033[{code}m{s}\033[0m" if COLOR else str(s)


bold, dim, red, yellow, magenta, cyan = (style(c) for c in ("1", "2", "31", "33", "35", "36"))


def width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


def vlen(s: str) -> int:
    return len(ANSI.sub("", s))


def fit(parts: list[str], gap: str = "   ") -> str:
    # drop parts off the end until the line fits, a wrapped line breaks the redraw
    parts = [p for p in parts if p]
    while len(parts) > 1 and vlen(gap.join(parts)) > width() - 1:
        parts = parts[:-1]
    return gap.join(parts)


def clip(text: str, w: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= w else text[: max(w - 1, 0)] + "…"


def bar(frac: float, n: int, color=cyan) -> str:
    full = round(min(max(frac, 0.0), 1.0) * n)
    return color("━" * full) + dim("─" * (n - full))


def num(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def clock(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


THERMAL = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def thermal(t: int | None) -> str:
    if t is None:
        return dim("-")
    return (dim, yellow, red, lambda s: bold(red(s)))[t](THERMAL[t])


# braille dot bits, [row][col] inside one 2x4 cell
DOTS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))


def chart(points: list[tuple[int, float]], w: int, h: int) -> tuple[list[str], float, float]:
    """points are (x, y) with x in 0..2w-1. returns h lines of braille plus the y range used."""
    grid = [[0] * w for _ in range(h)]
    if not points:
        return ["⠀" * w] * h, 0.0, 0.0
    ys = [y for _, y in points]
    # ignore the first few points for the scale, otherwise the starting loss squashes everything else flat
    hi = max(ys[len(ys) // 20 :] if len(ys) > 20 else ys)
    lo = min(ys)
    span = (hi - lo) or 1.0
    tall = h * 4 - 1
    prev = None
    for x, y in points:
        py = round((hi - min(y, hi)) / span * tall)
        # join to the previous point when they're neighbours so the line doesn't break up
        joined = prev is not None and x - prev[0] == 1
        for yy in range(min(py, prev[1]), max(py, prev[1]) + 1) if joined else (py,):
            grid[yy // 4][x // 2] |= DOTS[yy % 4][x % 2]
        prev = (x, py)
    return ["".join(chr(0x2800 + c) for c in row) for row in grid], lo, hi


def readline_safe(s: str) -> str:
    # colour codes in an input() prompt need wrapping or readline gets the width wrong
    return ANSI.sub(lambda m: f"\001{m.group()}\002", s)


class Live:
    """A block of lines at the bottom of the terminal that redraws in place. say() prints above it."""

    def __init__(self):
        self.lines: list[str] = []
        self.last = 0.0

    def due(self) -> bool:
        return TTY and time.time() - self.last > 0.1

    def _clear(self):
        if self.lines:
            sys.stdout.write("\r" + "\033[1A" * (len(self.lines) - 1) + "\033[J")

    def show(self, lines: list[str] | str):
        if not TTY:
            return
        lines = [lines] if isinstance(lines, str) else lines
        self._clear()
        sys.stdout.write("\n".join(lines))
        sys.stdout.flush()
        self.lines, self.last = lines, time.time()

    def say(self, text: str = ""):
        if TTY:
            self._clear()
        print(text, flush=True)
        if TTY and self.lines:
            sys.stdout.write("\n".join(self.lines))
            sys.stdout.flush()

    def done(self, text: str | None = None):
        if TTY:
            self._clear()
            sys.stdout.flush()
        self.lines = []
        if text is not None:
            print(text, flush=True)
