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
ASSETS = ROOT / "scripts" / "report_assets"
DOCS = ((ASSETS / "report.md", ROOT / "REPORT.md"), (ASSETS / "readme.md", ROOT / "README.md"))
BANNER = ("<!-- Generated from scripts/report_assets/{name} by scripts/render_report.py — edit the\n"
          "     template, not this file. Every figure below is resolved from reports/ at render time. -->\n")
PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def render(tmpl, reg):
    text = tmpl.read_text()
    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(text)} - set(reg))
    if missing:
        raise SystemExit(f"{tmpl.name}: unresolved placeholders (add them to report_numbers.build): "
                         f"{', '.join(missing)}")
    return BANNER.format(name=tmpl.name) + PLACEHOLDER.sub(lambda m: reg[m.group(1)], text)


HEADING = re.compile(r"^#{1,4} .+$", re.M)
FENCE = re.compile(r"^```.*?^```", re.M | re.S)          # shell comments inside a fence are not headings
NUMBERISH = re.compile(r"[−+-]?\d[\d,.]*[%×x]?")
WILDCARD = "\x00"


def _flat(block):
    return re.sub(r"\s+", " ", block.replace("*", "")).strip()


def _pattern(block):
    """A template block as a regex: placeholders and figures become wildcards, everything else is fixed.

    Matching this way is what keeps the guard honest. A registry value can be a number (3.58) or a word
    ("seven", "months-in-profit under 80%"), and either changes the rendered sentence without anyone
    having edited it — comparing rendered text to rendered text would call that a lost paragraph. The
    template's own wording is the invariant, so the check is: does every paragraph in the file still
    match some paragraph the template can produce?"""
    marked = NUMBERISH.sub(WILDCARD, PLACEHOLDER.sub(WILDCARD, _flat(block)))
    parts = re.split(f"({WILDCARD})", marked)
    # a template block that is nothing but a placeholder (a generated table) would compile to `.*?` and
    # match every paragraph in the file, which would silently disable the whole guard
    if sum(len(x) for x in parts if x != WILDCARD) < 20:
        return None
    return re.compile("".join(".*?" if part == WILDCARD else re.escape(part) for part in parts)
                      + r"\Z", re.S)


def _guard_dropped_content(tmpl, out, new):
    """Refuse to overwrite prose that exists in the generated file but not in what we are about to write.

    This is the one way a generated document loses work: someone edits REPORT.md — the obvious place,
    it is the document — and the next render replaces the file. It has cost four sections already, so
    headings are checked, and paragraphs too, because a row added to a table or a paragraph added to an
    existing section vanishes just as quietly and is harder to notice afterwards.

    Comparison is on the *wording* with figures stripped: a number that moved because an artifact moved
    is not a lost paragraph, but a sentence nobody wrote in the template is. Generated tables are
    exempt — their rows legitimately appear and disappear with the data. `--allow-drop` overrides, which
    is what a deliberate deletion looks like."""
    if not out.exists() or "--allow-drop" in sys.argv:
        return
    current = out.read_text()
    headings = (lambda t: set(HEADING.findall(FENCE.sub("", t))))
    dropped_h = [h for h in headings(current) if h not in headings(new)]
    patterns = [p for p in (_pattern(b) for b in tmpl.read_text().split("\n\n") if b.strip())
                if p is not None]
    dropped_p = [b for b in current.split("\n\n")
                 if b.strip() and not b.lstrip().startswith(("|", "<!--"))   # tables churn; banner is ours
                 and len(_flat(b)) > 80                                      # ignore fragments and captions
                 and not any(p.match(_flat(b)) for p in patterns)]
    if not (dropped_h or dropped_p):
        return
    lines = [f"{out.name}: rendering would drop content that is in the file but not in "
             f"{tmpl.relative_to(ROOT)}:"]
    for h in dropped_h:
        lines.append(f"    section  {h}")
    for b in dropped_p:
        lines.append(f"    text     {_flat(b)[:96]}…")
    lines += ["  This was almost certainly written straight into the generated file. Move it into the",
              "  template (that is the editable copy), then render again. If the removal is deliberate:",
              "    python scripts/render_report.py --allow-drop"]
    raise SystemExit("\n".join(lines))


def main():
    reg = build()
    check = "--check" in sys.argv
    for tmpl, out in DOCS:
        new = render(tmpl, reg)
        _guard_dropped_content(tmpl, out, new)
        if check:
            if (out.read_text() if out.exists() else "") != new:
                raise SystemExit(f"{out.name} is stale — the artifacts moved since it was rendered.\n"
                                 f"  fix: python scripts/render_report.py")
            print(f"{out.name} is current with the artifacts")
            continue
        out.write_text(new)
        print(f"{out.name} <- {tmpl.relative_to(ROOT)} "
              f"({len(PLACEHOLDER.findall(tmpl.read_text()))} figures resolved)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
