#!/usr/bin/env python3
"""Check that every relative link in the Markdown docs (and llms.txt) points to an existing file or heading, and that
llms-full.txt matches what rdna3/tools/make-llms-full.sh would write. Exit 1 on any problem."""
import pathlib
import re
import subprocess
import tempfile
import sys

root = pathlib.Path(__file__).resolve().parents[2]
files = [root / "README.md", root / "llms.txt", root / "AGENTS.md", root / "CONTRIBUTING.md", root / "NOTICE",
         root / "SECURITY.md", root / "CHANGELOG.md",
         *sorted((root / "docs" / "rdna3").glob("*.md")), root / "rdna3" / "README.md"]


def anchors(md: pathlib.Path) -> set[str]:
    out = set()
    for line in md.read_text().splitlines():
        if line.startswith("#"):
            slug = re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
            out.add(slug)
    return out


bad = []
for f in files:
    if not f.exists():
        continue
    for target in re.findall(r"\]\(([^)\s]+)\)", f.read_text()):
        if re.match(r"[a-z]+://|mailto:", target):
            continue
        path, _, frag = target.partition("#")
        dest = (f.parent / path).resolve() if path else f
        if not dest.exists():
            bad.append(f"{f.relative_to(root)}: missing {target}")
        elif frag and dest.suffix == ".md" and frag not in anchors(dest):
            bad.append(f"{f.relative_to(root)}: no heading #{frag} in {path or f.name}")
full = root / "llms-full.txt"
with tempfile.TemporaryDirectory() as tmp:  # regenerate aside and compare: a check never rewrites tracked files
    fresh = pathlib.Path(tmp) / "llms-full.txt"
    subprocess.run([str(root / "rdna3/tools/make-llms-full.sh"), str(fresh)], check=True, capture_output=True)
    if not full.exists() or full.read_text() != fresh.read_text():
        bad.append("llms-full.txt is stale: run rdna3/tools/make-llms-full.sh and commit it")
print("\n".join(bad) or f"docs ok ({len(files)} files)")
sys.exit(1 if bad else 0)
