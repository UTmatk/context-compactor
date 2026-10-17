#!/usr/bin/env python3
# Copyright (C) 2026 UTmatk
# SPDX-License-Identifier: GPL-3.0-or-later
"""Self-test for kb-skill (no pytest required).

    python tests/selftest.py            # run everything that needs tiktoken
    python tests/selftest.py --quick    # skip subprocess/CLI checks

Each check maps to a bug from the review, so a regression fails loudly.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")

failures: list[str] = []
checks = 0


def load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check(name: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}" + (f"  [{detail}]" if detail else ""))
        failures.append(name)


def write(tmp: str, name: str, text: str) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


CN = ("大型语言模型是一种基于神经网络的语言模型。上下文窗口决定大型语言模型一次能处理多少文本。"
      "训练数据决定大型语言模型的神经网络能力。大型语言模型的上下文窗口越大，训练数据能塞进去的越多。"
      "神经网络是大型语言模型的核心，训练数据决定神经网络学到什么。上下文窗口限制了神经网络一次能看到的"
      "训练数据，所以大型语言模型需要更大的上下文窗口来消化更多训练数据。")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip subprocess CLI checks")
    args = ap.parse_args()

    compact = load("kb_compact", os.path.join(SCRIPTS, "compact.py"))
    auto = load("kb_auto", os.path.join(SCRIPTS, "auto_compactor.py"))

    if not compact.HAVE_TOK:
        print("tiktoken missing — install it first:  pip install tiktoken")
        return 2

    tmp = tempfile.mkdtemp(prefix="kb_selftest_")

    print("[1] legend parsing must not read the compressed body")
    packed = write(tmp, "packed.txt",
                   "Dictionary (expand symbols before use):\n"
                   "§ = 大型语言模型\n¶ = 上下文窗口\n"
                   "\n"
                   "body: value = result\n"
                   "train = fit\n")
    m = auto.load_dictionary(packed)
    check("no body lines become entries", set(m) == {"大型语言模型", "上下文窗口"}, repr(m))
    check("compact.parse_legend agrees", compact.parse_legend(open(packed, encoding="utf-8").read()) == m)

    print("[2] bare legend (no header, no blank line)")
    bare = write(tmp, "bare.txt", "# a comment\n§ = 大型语言模型\n← = 训练数据\n")
    m2 = auto.load_dictionary(bare)
    check("bare legend parsed", m2 == {"大型语言模型": "§", "训练数据": "←"}, repr(m2))

    print("[3] phrase containing '=' survives")
    eq = write(tmp, "eq.txt", "§ = a = b\n")
    check("phrase with '='", auto.load_dictionary(eq) == {"a = b": "§"}, repr(auto.load_dictionary(eq)))

    print("[4] JSON orientation auto-detect")
    fwd = write(tmp, "fwd.json", '{"大型语言模型": "§", "训练数据": "←"}')
    rev = write(tmp, "rev.json", '{"§": "大型语言模型", "←": "训练数据"}')
    check("phrase->symbol", auto.load_dictionary(fwd) == {"大型语言模型": "§", "训练数据": "←"})
    check("symbol->phrase", auto.load_dictionary(rev) == {"大型语言模型": "§", "训练数据": "←"})

    print("[5] symbol pool: every usable symbol is exactly 1 token")
    bad = [c for c in auto.usable_symbols() if len(auto.ENC.encode(c)) != 1]
    check("all 1 token", not bad, repr(bad))
    check(f"pool big enough for --top 15 (got {len(auto.usable_symbols())})", len(auto.usable_symbols()) >= 15)

    print("[6] audit warns about symbols that are not 1 token")
    legacy = {"大型语言模型": "§", "训练数据": "◆"}
    check("2-token symbol reported", any(">1" in w for w in auto.audit_dictionary(legacy)),
          repr(auto.audit_dictionary(legacy)))
    check("shipped test_dict is clean", not auto.audit_dictionary(
        auto.load_dictionary(os.path.join(ROOT, "test_dict.txt"))))

    print("[7] shifted-window fragments are rejected (A/B parity)")
    mc, lc, _ = compact.build_mapping(CN, top=15, cjk_max=6)
    ma, la = auto.build_mapping(CN, top=15, cjk_max=6)
    check("compact == auto mapping", mc == ma, f"{mc} != {ma}")
    check("compact == auto legend", lc == la)
    frag = [p for p in mc if any(q != p and compact.cjk_overlap(p, q) for q in mc)]
    check("no overlapping CJK fragments chosen", not frag, repr(frag))

    # position-level guard: 「训练数据」被选中后，不得再选跨边界拼出来的「练数据决定」
    spans = {p: [(x.start(), x.end()) for x in compact.phrase_regex(p).finditer(CN)] for p in mc}
    pos_conflict = []
    for p in mc:
        for q in mc:
            if p >= q:
                continue
            if any(a < y and x < b for a, b in spans[p] for x, y in spans[q]):
                pos_conflict.append((p, q))
    check("no chosen phrase overlaps another in the source text", not pos_conflict, repr(pos_conflict))

    print("[8] compress/expand round-trip")
    comp = compact.compress(CN, mc)
    check("expand(compress(x)) == x", compact.expand(comp, mc) == CN)
    check("text actually shrank", compact.n_tokens(comp) < compact.n_tokens(CN))

    print("[9] missing tiktoken must raise, not silently return {}/0%")
    saved_enc, saved_flag = compact.ENC, compact.HAVE_TOK
    try:
        compact.ENC, compact.HAVE_TOK = None, False
        compact._symbol_cache = None
        try:
            compact.build_mapping(CN)
            raised = False
        except compact.TokenizerMissing:
            raised = True
        check("TokenizerMissing raised", raised)
    finally:
        compact.ENC, compact.HAVE_TOK = saved_enc, saved_flag
        compact._symbol_cache = None

    print("[10] transport failure rolls the turn back")
    class Boom:
        def __call__(self, messages, **kw):
            raise RuntimeError("upstream down")

    sess = auto.AutoCompactSession(transport=Boom())
    try:
        sess.chat("你好")
    except RuntimeError:
        pass
    check("history has no orphan turn", sess.history == [], repr(sess.history))

    print("[11] fixed-dictionary mode still saves (~18.6% on the demo corpus)")
    fixed = write(tmp, "fixed.txt",
                  "Dictionary (expand symbols before use):\n"
                  "§ = 大型语言模型\n¶ = 上下文窗口\n← = 训练数据\n● = 神经网络\n")
    s = auto.AutoCompactSession(predefined_dict=fixed)
    s.system = "You are a helpful assistant."
    for t in range(4):
        s.chat(auto.DEMO_BASE + f"（这是第 {t + 1} 轮提问）")
    st = s.stats()
    check(f"4-round fixed dict in 15..20% (got {st['saved_pct']:.1f}%)", 15.0 <= st["saved_pct"] <= 20.0,
          str(st))

    print("[12] default rebuild mode still wins on a long chat")
    s2 = auto.AutoCompactSession()
    s2.system = "You are a helpful assistant."
    for t in range(6):
        s2.chat(auto.DEMO_BASE + f"（这是第 {t + 1} 轮提问）")
    st2 = s2.stats()
    check(f"6-round cumulative > 30% (got {st2['saved_pct']:.1f}%)", st2["saved_pct"] > 30.0, str(st2))
    legend2 = s2._build_active_mapping()[1]
    chosen2 = [ln.split(" = ", 1)[1] for ln in legend2]
    frag2 = [p for p in chosen2 if any(q != p and auto.cjk_overlap(p, q) for q in chosen2)]
    check("dictionary has no overlapping entries", not frag2, repr(legend2))

    print("[13] UTF-8 console survives symbols on CP936")
    check("stdout encoding is utf-8", (sys.stdout.encoding or "").lower().replace("-", "") == "utf8",
          sys.stdout.encoding)

    if not args.quick:
        print("[14] CLI end-to-end: compress -> expand restores the original")
        src = write(tmp, "src.txt", CN)
        out = os.path.join(tmp, "out.txt")
        back = os.path.join(tmp, "back.txt")
        r1 = subprocess.run([sys.executable, os.path.join(SCRIPTS, "compact.py"),
                             "--input", src, "--output", out, "--stats"],
                            capture_output=True, text=True, encoding="utf-8")
        check("compress exit 0", r1.returncode == 0, r1.stderr[-400:])
        r2 = subprocess.run([sys.executable, os.path.join(SCRIPTS, "compact.py"),
                             "--input", out, "--output", back, "--expand"],
                            capture_output=True, text=True, encoding="utf-8")
        check("expand exit 0", r2.returncode == 0, r2.stderr[-400:])
        restored = open(back, encoding="utf-8").read().strip()
        check("CLI round-trip byte-identical", restored == CN, restored[:80])

        print("[15] root shim re-exports the canonical implementation")
        shim_code = (
            "import auto_compactor as m;"
            "print(m.AutoCompactSession.__module__);"
            "raise SystemExit(m.main(['--rounds','2','--dict',r'%s']))" % fixed
        )
        r3 = subprocess.run([sys.executable, "-c", shim_code],
                            capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
        check("shim import works", r3.returncode == 0 and "_kb_auto_compactor_impl" in r3.stdout,
              r3.stderr[-400:])

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
