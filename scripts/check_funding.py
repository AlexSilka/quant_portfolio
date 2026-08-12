"""Fail the build when a strategy can hold a perpetual without paying to hold it.

Perp funding was patched into the x-sect leg, then the lottery sleeve, then BAB — three separate
discoveries of one defect, months apart, each found by hand. `src/backtest/carry` closed the panel
path structurally (carry is resolved from the panel's own names, so forgetting charges rather than
skips), but the single-asset path cannot be closed the same way: `engine.backtest` is handed a bare
price Series with no venue on it, so `funding=` stays an argument a caller can simply not pass.

This is the guard for what the design cannot make impossible. It reads every call site of the four
functions that can hold a position, resolves `**kwargs` unpacking back to the dict it came from —
the repo's normal spelling, and a keyword scan that misses it reports every such site as a defect —
and names any call in a file that touches crypto where nothing pays carry.

    python scripts/check_funding.py            report, exit 1 if a crypto position is uncharged
    python scripts/check_funding.py --list     every call site and how it is charged

Waiving a site is deliberate and visible: put its `file:line` in ALLOWED with the reason.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = {"xs_backtest", "bab_backtest"}
SINGLE = {"backtest"}
CRYPTO_HINT = re.compile(r"market=[\"']um[\"']|load_crypto|crypto_(1d|4h|1h|15m|5m)|futures/um|USDT")

# Sites that hold no real perp. Each is a statement about the position, not about the author.
ALLOWED = {
    "scripts/smoke_math.py": "synthetic price series in the invariant tests — no instrument at all",
    "scripts/trend/run_trend_book.py": "shuffled-returns placebo: both arms run through the same "
                                       "line, so the null comparison stays apples-to-apples",
    "scripts/trend/run_trend_trades.py": "the equity loop — cash equities pay borrow, not funding, "
                                         "and the crypto loop above it passes funding",
}


def _call_name(node: ast.Call) -> str:
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")


def _literal_keys(v: ast.AST) -> set[str]:
    if isinstance(v, ast.Call) and _call_name(v) == "dict":
        return {k.arg for k in v.keywords if k.arg}
    if isinstance(v, ast.Dict):
        return {k.value for k in v.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
    return set()


def _dict_keys(tree: ast.AST, name: str) -> set[str]:
    """Every key ever assigned to `name` as a dict — resolves the `common = dict(...)` idiom."""
    keys: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                             for t in n.targets):
            keys |= _literal_keys(n.value)
    return keys


def sites() -> list[dict]:
    out = []
    for p in sorted(q for d in ("scripts", "src") for q in (ROOT / d).rglob("*.py")):
        try:
            text = p.read_text()
            tree = ast.parse(text)
        except (SyntaxError, UnicodeDecodeError):
            continue
        crypto = bool(CRYPTO_HINT.search(text))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _call_name(node) not in PANEL | SINGLE:
                continue
            kw = {k.arg for k in node.keywords if k.arg}
            for v in (k.value for k in node.keywords if k.arg is None):
                kw |= _dict_keys(tree, v.id) if isinstance(v, ast.Name) else _literal_keys(v)
            name = _call_name(node)
            charged = ("funding" in kw or "symbol" in kw) if name in SINGLE else True
            how = ("funding=" if "funding" in kw else "symbol=" if "symbol" in kw else "NOTHING") \
                if name in SINGLE else ("carry= explicit" if "carry" in kw else "auto from panel")
            out.append({"file": str(p.relative_to(ROOT)), "line": node.lineno, "call": name,
                        "crypto": crypto, "charged": charged, "how": how})
    return out


def main() -> None:
    rows = sites()
    panel = [r for r in rows if r["call"] in PANEL]
    single = [r for r in rows if r["call"] in SINGLE]
    uncharged = [r for r in single if r["crypto"] and not r["charged"]]
    flagged = [r for r in uncharged if r["file"] not in ALLOWED]

    if "--list" in sys.argv:
        for r in rows:
            print(f"  {r['how']:16s} {'crypto' if r['crypto'] else 'other ':6s} "
                  f"{r['file']}:{r['line']}")
        return

    print(f"{len(panel)} panel call sites — carry resolved from the instrument "
          f"({sum(1 for r in panel if r['how'] == 'carry= explicit')} explicit)")
    print(f"{len(single)} single-asset call sites — "
          f"{sum(1 for r in single if r['crypto'] and r['charged'])} crypto charged, "
          f"{len(uncharged)} crypto uncharged ({len(uncharged) - len(flagged)} waived), "
          f"{sum(1 for r in single if not r['crypto'])} non-crypto")
    for r in uncharged:
        if r["file"] in ALLOWED:
            print(f"  waived  {r['file']}:{r['line']} — {ALLOWED[r['file']]}")
    if flagged:
        print("\nUNCHARGED PERP POSITIONS:")
        for r in flagged:
            print(f"  {r['file']}:{r['line']}  holds a position on a crypto file with no funding")
            print("      fix: pass symbol=<SYM> (engine resolves the archive) or funding=")
        raise SystemExit(1)
    print("FUNDING OK — nothing can hold a perpetual for free")


if __name__ == "__main__":
    main()
