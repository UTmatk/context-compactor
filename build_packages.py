#!/usr/bin/env python3
# Copyright (C) 2026 UTmatk
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build the two distributable skill zips deterministically.

    python build_packages.py [--out DIR]

Each archive is self-contained (SKILL.md + scripts/ + LICENSE, because the
project is GPL-3.0-or-later and a zip without the licence text is not a valid
redistribution). Entry timestamps are fixed so rebuilding produces identical
bytes.
"""
from __future__ import annotations

import argparse
import os
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
FIXED_TIME = (2026, 1, 1, 0, 0, 0)

PACKAGES = [
    ("context-compactor", [("context-compactor/SKILL.md", "SKILL.md"),
                           ("scripts/compact.py", "scripts/compact.py")]),
    ("context-auto-compactor", [("context-auto-compactor/SKILL.md", "SKILL.md"),
                                ("scripts/auto_compactor.py", "scripts/auto_compactor.py")]),
]
COMMON = [("scripts/requirements.txt", "scripts/requirements.txt"),
          ("LICENSE.txt", "LICENSE")]


def build(name: str, members: list[tuple[str, str]], out_dir: str) -> str:
    path = os.path.join(out_dir, name + ".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_rel, arc in sorted(members + COMMON):
            src = os.path.join(ROOT, src_rel)
            if not os.path.exists(src):
                raise SystemExit(f"missing source file: {src}")
            info = zipfile.ZipInfo(f"{name}/{arc}", date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(src, "rb") as f:
                zf.writestr(info, f.read())
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=ROOT, help="output directory (default: repo root)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, members in PACKAGES:
        path = build(name, members, args.out)
        print(f"built {path} ({os.path.getsize(path)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
