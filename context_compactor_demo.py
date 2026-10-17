#!/usr/bin/env python3
# Copyright (C) 2026 UTmatk
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prompt dictionary compaction demo.

这是**教学用的薄壳**：真正的算法只有一份，在 `scripts/compact.py`。历史教训是
这里曾经复制过一份早期实现，结果 `compact.py` 修好了、demo 没修，两边选词行为
不一样。现在 demo 直接调用规范实现，两者不可能漂移。
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TARGET = os.path.join(_HERE, "scripts", "compact.py")

_spec = importlib.util.spec_from_file_location("_kb_compact_impl", _TARGET)
if _spec is None or _spec.loader is None:                       # pragma: no cover
    raise SystemExit(f"canonical implementation missing: {_TARGET}")
compact = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = compact
_spec.loader.exec_module(compact)


# Repeat a base paragraph several times to simulate a long, jargon-heavy
# document / multi-turn chat where the same terms recur constantly.
BASE = (
    "A large language model is a large language model that learns from training data. "
    "Machine learning helps a large language model improve. The context window of a large "
    "language model limits how much training data it can see. We use machine learning to train "
    "a large language model on training data. The context window grows but training data is huge. "
    "A neural network is the core of a large language model. Machine learning updates the neural "
    "network using training data. The context window must fit the neural network and the training "
    "data. Many teams build a large language model with machine learning and a big context window."
)
SAMPLE = " ".join([BASE] * 6)


def main() -> int:
    text = SAMPLE
    mapping, legend_lines, legend_tok = compact.build_mapping(
        text, top=15, max_ngram=3, cjk_min=2, cjk_max=6)
    compressed = compact.compress(text, mapping)

    orig_tok = compact.n_tokens(text)
    comp_only_tok = compact.n_tokens(compressed)
    total_tok = compact.n_tokens(compressed) + legend_tok
    per_turn = orig_tok - comp_only_tok
    break_even = (legend_tok / per_turn) if per_turn > 0 else float("inf")

    print("=" * 64)
    print("PROMPT DICTIONARY COMPACTION DEMO (canonical implementation)")
    print("tokenizer :", "tiktoken cl100k_base" if compact.HAVE_TOK else "proxy (~4 chars/tok)")
    print("=" * 64)
    print(f"original tokens            : {orig_tok}")
    print(f"compressed body tokens     : {comp_only_tok}")
    print(f"dictionary (legend) tokens : {legend_tok}  (sent ONCE)")
    print(f"total with dictionary      : {total_tok}")
    print(f"tokens saved               : {orig_tok - total_tok}  ({100 * (orig_tok - total_tok) / orig_tok:.1f}%)")
    if break_even != float("inf"):
        print(f"break-even after           : {break_even:.2f} repeated turns")
    print("-" * 64)
    print("DICTIONARY:")
    for line in legend_lines:
        print("  " + line)
    print("-" * 64)
    print("COMPRESSED (head):")
    print(compressed[:460])
    print("-" * 64)
    restored = compact.expand(compressed, mapping)
    print("round-trip restore         :", "OK (byte-identical)" if restored == text else "MISMATCH")
    return 0 if restored == text else 1


if __name__ == "__main__":
    raise SystemExit(main())
