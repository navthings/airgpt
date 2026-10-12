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


bold, dim, faint, red, green, yellow, cyan = (style(c) for c in ("1", "2", "2;3", "31", "32", "33", "36"))


def width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


def vlen(s: str) -> int:
    return len(ANSI.sub("", s))


def fit(parts: list[str], gap: str = "   ") -> str:
    # drop parts off the end until the line fits, so the status line never wraps
    while len(parts) > 1 and vlen(gap.join(parts)) > width() - 1:
        parts = parts[:-1]
    return gap.join(parts)


def clip(text: str, w: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= w else text[: max(w - 1, 0)] + "…"


def bar(frac: float, n: int = 28) -> str:
    full = round(min(max(frac, 0.0), 1.0) * n)
    return cyan("━" * full) + dim("━" * (n - full))


def num(n: float) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


def clock(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02}:{s:02}" if h else f"{m}:{s:02}"


def label(name: str, text: str = "") -> str:
    return f"  {bold(name.ljust(11))}{text}"


THERMAL = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


def thermal(t: int | None, pad: int = 0) -> str:
    if t is None:
        return dim("-".ljust(pad))
    return (dim, yellow, red, lambda s: bold(red(s)))[t](THERMAL[t].ljust(pad))


def readline_safe(s: str) -> str:
    # input() prompts need colour codes wrapped so readline counts the width right
    return ANSI.sub(lambda m: f"\001{m.group()}\002", s)


class Live:
    """One status line that redraws in place. say() prints a normal line above it."""

    def __init__(self):
        self.line, self.last = "", 0.0

    def show(self, line: str, force: bool = False):
        self.line = line
        if TTY and (force or time.time() - self.last > 0.1):
            sys.stdout.write("\r\033[2K" + line)
            sys.stdout.flush()
            self.last = time.time()

    def say(self, text: str = ""):
        if TTY and self.line:
            sys.stdout.write("\r\033[2K")
        print(text, flush=True)
        if TTY and self.line:
            sys.stdout.write(self.line)
            sys.stdout.flush()

    def done(self, text: str | None = None):
        if TTY and self.line:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
        self.line = ""
        if text is not None:
            print(text, flush=True)
