"""Write REPORT.md from its prose template plus the measured numbers.

    python scripts/render_report.py          # rebuild REPORT.md
    python scripts/render_report.py --check  # fail if the committed REPORT.md is not what renders now

The prose lives in `scripts/report_assets/report.md`; every figure it quotes is a `{{name}}` resolved by
`scripts/report_numbers.py` from the committed artifacts. Edit the template, never REPORT.md. `--check`
is the gate that makes that stick: it is wired into `make lint`, so a book re-run that leaves the report
behind fails the build instead of shipping a report that argues with its own artifacts.

An unresolved `{{name}}` is an error, not a blank: a number nothing can resolve is a number nobody
measured."""
import re
import sys
from pathlib import Path

from report_numbers import build

ROOT = Path(__file__).resolve().parents[1]
TMPL = ROOT / "scripts" / "report_assets" / "report.md"
OUT = ROOT / "REPORT.md"
BANNER = ("<!-- Generated from scripts/report_assets/report.md by scripts/render_report.py — edit the\n"
          "     template, not this file. Every figure below is resolved from reports/ at render time. -->\n")
PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def render():
    reg = build()
    text = TMPL.read_text()
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(text)} - set(reg))
    if missing:
        raise SystemExit(f"unresolved placeholders (add them to report_numbers.build): {', '.join(missing)}")
    return BANNER + PLACEHOLDER.sub(lambda m: reg[m.group(1)], text)


def main():
    new = render()
    if "--check" in sys.argv:
        current = OUT.read_text() if OUT.exists() else ""
        if current != new:
            raise SystemExit("REPORT.md is stale — the artifacts moved since it was rendered.\n"
                             "  fix: python scripts/render_report.py")
        print("REPORT.md is current with the artifacts")
        return
    OUT.write_text(new)
    print(f"REPORT.md <- {TMPL.relative_to(ROOT)} ({len(PLACEHOLDER.findall(TMPL.read_text()))} figures resolved)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
