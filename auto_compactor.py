#!/usr/bin/env python3
# Copyright (C) 2026 UTmatk
# SPDX-License-Identifier: GPL-3.0-or-later
"""兼容垫片（backward-compatible shim）。

规范实现只有一份：`scripts/auto_compactor.py`。此文件保留是为了让在项目根目录
下工作的旧代码（`from auto_compactor import AutoCompactSession`、`python
auto_compactor.py`）继续可用，同时不再产生"根目录一份 + zip 里一份"的算法漂移。

不要在这里写实现。
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "scripts", "auto_compactor.py")

if not os.path.exists(_TARGET):                      # pragma: no cover
    raise ImportError(
        f"canonical implementation missing: {_TARGET}\n"
        "restore scripts/auto_compactor.py from the skill package")

_spec = importlib.util.spec_from_file_location("_kb_auto_compactor_impl", _TARGET)
assert _spec and _spec.loader
_impl = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _impl
_spec.loader.exec_module(_impl)

# re-export the public surface
for _name, _value in vars(_impl).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

__all__ = [n for n in vars(_impl) if not n.startswith("_")]


if __name__ == "__main__":
    raise SystemExit(_impl.main())
