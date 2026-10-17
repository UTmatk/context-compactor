#!/usr/bin/env python3
# Copyright (C) 2026 UTmatk
# SPDX-License-Identifier: GPL-3.0-or-later
"""context-compactor: deterministic dictionary compression for prompts.

Maps high-frequency phrases (English word groups or CJK character runs) to
single-token symbols so a long text / conversation history fits in fewer tokens
before being sent to a model. Deterministic and reversible: the mapping is a
fixed table that can be expanded exactly with ``expand()``.

Usage:
    python scripts/compact.py --input long.txt --output packed.txt --stats
    cat long.txt | python scripts/compact.py --stats
    python scripts/compact.py --expand --input packed.txt      # restore

Requires tiktoken for exact token counts. Without it the tool refuses to build
a dictionary instead of silently producing a useless one (see --allow-proxy).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter

# ─────────────────────────── 环境 / 编码 ───────────────────────────

def _force_utf8_stdio() -> None:
    """Windows 控制台常为 GBK/CP936，打印 § ¶ ● 之流会 UnicodeEncodeError。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


_force_utf8_stdio()


class TokenizerMissing(RuntimeError):
    """tiktoken is not installed; exact counting (and dictionary selection) is off."""


try:                                                    # noqa: SIM105
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    HAVE_TOK = True
except Exception:                                       # pragma: no cover
    tiktoken = None
    ENC = None
    HAVE_TOK = False


def n_tokens(s: str) -> int:
    """Token count. Falls back to a crude ASCII proxy when tiktoken is absent."""
    if ENC is not None:
        return len(ENC.encode(s))
    return max(1, len(s) // 4)          # ~4 chars/token; meaningless for CJK


def require_tokenizer(allow_proxy: bool = False) -> None:
    """Refuse to build a dictionary without exact token counts.

    The proxy counts every Chinese character as 1/4 token, which makes every
    CJK phrase fail the ``n_tokens(phrase) >= 2`` test — i.e. without tiktoken
    the compressor silently selects nothing and reports "0% saved".
    """
    if ENC is not None or allow_proxy:
        return
    raise TokenizerMissing(
        "tiktoken is required for exact token counting.\n"
        "  pip install tiktoken\n"
        "  (pass --allow-proxy to see a *non-representative* estimate anyway)"
    )


# ─────────────────────────── 符号池 ───────────────────────────
# Ordered by "least likely to appear in natural text". Every entry is verified
# to encode as exactly 1 token at runtime (many emoji / geometric shapes are
# 2+ tokens in cl100k_base and would make compression a net loss).
SYMBOL_POOL = (
    "§¶●★☆♥♀♪"          # original 8, kept first so old dictionaries stay valid
    "←↑→↓"              # arrows
    "─━│█░"             # box drawing / blocks
    "†¬±"               # safe punctuation-ish
    "αβγδεηθκλμπρστυφχω"  # Greek; unambiguous vs Latin, 1 token each
)

_symbol_cache: list[str] | None = None


def usable_symbols() -> list[str]:
    """Symbols from SYMBOL_POOL that really cost 1 token (cached)."""
    global _symbol_cache
    if _symbol_cache is None:
        if ENC is None:
            _symbol_cache = list(SYMBOL_POOL)
        else:
            _symbol_cache = [c for c in SYMBOL_POOL if n_tokens(c) == 1]
    return _symbol_cache


def pick_symbols(n: int) -> list[str]:
    return usable_symbols()[:n]


def is_one_token_symbol(ch: str) -> bool:
    return ENC is None or n_tokens(ch) == 1


# ─────────────────────────── 语料切分 ───────────────────────────

def latin_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*", text)


def cjk_runs(text: str) -> list[str]:
    return re.findall(r"[一-鿿]+", text)


def latin_ngrams(words: list[str], max_n: int) -> Counter:
    counts: Counter = Counter()
    n = len(words)
    for k in range(1, max_n + 1):
        for i in range(n - k + 1):
            counts[" ".join(words[i:i + k])] += 1
    return counts


def cjk_ngrams(run: str, min_l: int, max_l: int) -> Counter:
    counts: Counter = Counter()
    L = len(run)
    for k in range(min_l, max_l + 1):
        for i in range(L - k + 1):
            counts[run[i:i + k]] += 1
    return counts


def cjk_overlap(a: str, b: str) -> bool:
    """True if two CJK phrases share a >=4-char substring (shifted windows).

    Without this guard the greedy picker happily selects "型语言模型的" next to
    "大型语言模型": a wasted dictionary slot that also makes the legend noisy.
    """
    if not (re.search(r"[一-鿿]", a) and re.search(r"[一-鿿]", b)):
        return False
    if len(a) < 4 or len(b) < 4:
        return a in b or b in a
    for i in range(len(a) - 3):
        if a[i:i + 4] in b:
            return True
    return False


def overlaps(cand: str, chosen: str) -> bool:
    """Candidate is redundant with an already-chosen entry."""
    return cand in chosen or chosen in cand or cjk_overlap(cand, chosen)


def phrase_regex(phrase: str) -> re.Pattern:
    """Regex used to locate/replace a phrase (Latin gets word boundaries)."""
    if re.search(r"[A-Za-z0-9]", phrase):
        return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")
    return re.compile(re.escape(phrase))


def is_fragment(phrase: str, cnt: int, counts: Counter, text: str) -> bool:
    """True if `phrase` is only ever a slice of a longer *repeated* unit.

    「言模型是一种模型」的每次出现左边都是「语」，且更长的「语言模型是一种模型」
    出现次数完全相同 —— 说明它只是被窗口切下来的一截，选它等于把一条长词的槽位浪费
    在半个词上。判据用"扩展一格的串在 counts 里是否有同样多的出现次数"，因此
    被标点/边界截断的短语（扩展串不存在）不会被误杀。
    """
    occ = [(m.start(), m.end()) for m in re.finditer(re.escape(phrase), text)]
    if len(occ) < 2:
        return False
    if re.search(r"[A-Za-z0-9]", phrase):
        nxt = set(re.findall(re.escape(phrase) + r" ([A-Za-z0-9']+)", text))
        prv = set(re.findall(r"([A-Za-z0-9']+) " + re.escape(phrase), text))
        forms = [phrase + " " + w for w in nxt] + [w + " " + phrase for w in prv]
    else:
        lefts = {text[a - 1] if a > 0 else "" for a, _ in occ}
        rights = {text[b] if b < len(text) else "" for _, b in occ}
        forms = [next(iter(lefts)) + phrase] if len(lefts) == 1 and "" not in lefts else []
        if len(rights) == 1 and "" not in rights:
            forms.append(phrase + next(iter(rights)))
    return any(counts.get(form, 0) >= cnt for form in forms)


# ─────────────────────────── 建表 / 替换 / 还原 ───────────────────────────

def build_mapping(text: str, top: int = 15, max_ngram: int = 3,
                  cjk_min: int = 2, cjk_max: int = 6,
                  allow_proxy: bool = False):
    """Return (mapping {phrase: symbol}, legend lines, legend token cost)."""
    require_tokenizer(allow_proxy)
    counts: Counter = Counter()
    counts.update(latin_ngrams(latin_words(text), max_ngram + 1))
    for run in cjk_runs(text):
        counts.update(cjk_ngrams(run, cjk_min, cjk_max + 1))

    raw = []
    for phrase, cnt in counts.items():
        if re.search(r"[A-Za-z0-9]", phrase):
            words = phrase.split()
            if len(words) > max_ngram:
                continue                 # +1 headroom only feeds the fragment test
            if any(w[:1].isupper() for w in words[1:]):
                continue                 # spans a sentence boundary ("data The context")
        elif len(phrase) > cjk_max:
            continue
        tlen = n_tokens(phrase)
        if tlen < 2 or cnt < 2:          # 1-token or single occurrence: no gain
            continue
        raw.append((cnt * (tlen - 1), cnt, tlen, phrase))
    raw.sort(reverse=True)               # biggest savers first

    pool = usable_symbols()
    limit = min(top, len(pool))

    # 位置级重叠守卫：光看"短语是不是另一个短语的子串"不够。选中「训练数据」后，
    # 文本里剩下的文字会拼出「练数据决定」这种跨边界的滑动窗碎片——它既不是子串、
    # 也不共享 4 字窗口，但和已选短语在原文里重叠。这里直接比对出现位置跨度。
    spans: dict[str, list[tuple[int, int]]] = {}

    def span_of(phrase: str) -> list[tuple[int, int]]:
        if phrase not in spans:
            spans[phrase] = [(m.start(), m.end())
                             for m in phrase_regex(phrase).finditer(text)]
        return spans[phrase]

    def collides(cand: str, picked) -> bool:
        cand_spans = span_of(cand)
        for _, _, _, cp in picked:
            for a, b in cand_spans:
                for x, y in span_of(cp):
                    if a < y and x < b:
                        return True
        return False

    chosen = []
    for saving, cnt, tlen, phrase in raw:
        if any(overlaps(phrase, cp) for _, _, _, cp in chosen):
            continue                     # already covered by a longer entry
        if saving <= n_tokens("§ = " + phrase):
            continue                     # cannot pay for its own dictionary slot
        if is_fragment(phrase, cnt, counts, text):
            continue                     # only ever a slice of a longer repeated unit
        if chosen and collides(phrase, chosen):
            continue                     # shares text positions with a chosen phrase
        chosen.append((saving, cnt, tlen, phrase))
        if len(chosen) >= limit:
            break

    mapping: dict[str, str] = {}
    legend: list[str] = []
    legend_tok = 0
    for (saving, cnt, tlen, phrase), sym in zip(chosen, pick_symbols(len(chosen))):
        mapping[phrase] = sym
        line = f"{sym} = {phrase}"
        legend.append(line)
        legend_tok += n_tokens(line)
    return mapping, legend, legend_tok


def compress(text: str, mapping: dict[str, str]) -> str:
    """Replace phrases with symbols, longest phrase first (no nested replacement)."""
    for phrase in sorted(mapping, key=len, reverse=True):
        text = phrase_regex(phrase).sub(lambda _m, s=mapping[phrase]: s, text)
    return text


def expand(text: str, mapping: dict[str, str]) -> str:
    """Inverse of compress(): restore phrases from symbols."""
    reverse = {sym: phrase for phrase, sym in mapping.items()}
    for sym in sorted(reverse, key=len, reverse=True):
        text = text.replace(sym, reverse[sym])
    return text


# ─────────────────────────── 图例解析 ───────────────────────────

LEGEND_RE = re.compile(r"^(\S{1,4})\s*=\s*(.+)$")


def looks_like_symbol(s: str) -> bool:
    """A legend symbol: short, not alphanumeric/CJK, not an '=' or space."""
    return bool(s) and len(s) <= 4 and not re.search(r"[A-Za-z0-9_ =一-鿿]", s)


def parse_legend(text: str) -> dict[str, str]:
    """Parse a legend block into {phrase: symbol}.

    Stops at the first blank line after the block started, so the compressed
    body that follows a legend is never mistaken for dictionary entries
    (a line like ``train = fit`` used to become a bogus mapping).
    """
    mapping: dict[str, str] = {}
    started = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if started and mapping:
                break                     # end of the legend block
            continue
        low = line.lower()
        if low.startswith("dictionary") or low.startswith("expand"):
            started = True
            continue
        m = LEGEND_RE.match(line)
        if not m:
            if started and mapping:
                break                    # body begins
            continue
        sym, phrase = m.group(1).strip(), m.group(2).strip()
        if not looks_like_symbol(sym):
            if started and mapping:
                break
            continue
        if not phrase:
            continue
        mapping[phrase] = sym
        started = True
    return mapping


def strip_legend(text: str) -> str:
    """Return `text` without its leading legend block (used by --expand)."""
    lines = text.splitlines()
    started = False
    seen_entry = False
    cut = 0
    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            if started and seen_entry:
                cut = i + 1
                break
            continue
        low = line.lower()
        if low.startswith("dictionary") or low.startswith("expand"):
            started = True
            continue
        m = LEGEND_RE.match(line)
        if m and looks_like_symbol(m.group(1).strip()):
            started = True
            seen_entry = True
            continue
        if started and seen_entry:
            cut = i
            break
    return "\n".join(lines[cut:]).lstrip("\n")


def load_dictionary(path: str) -> dict[str, str]:
    """Load a dictionary from a legend file, a packed file, or a JSON file."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    if path.lower().endswith(".json"):
        import json
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            keys = list(data)
            if looks_like_symbol(str(keys[0])):       # {sym: phrase} -> invert
                return {str(p): str(s) for s, p in data.items()}
            return {str(p): str(s) for p, s in data.items()}
    return parse_legend(raw)


# ─────────────────────────── CLI ───────────────────────────

def _read_input(path: str | None) -> str:
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _write_output(path: str | None, text: str) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


def _emit_stats(text: str, mapping, legend, legend_tok, compressed: str, out: str) -> None:
    orig = n_tokens(text)
    comp = n_tokens(compressed)
    total = n_tokens(out)
    per = orig - comp
    be = (legend_tok / per) if per > 0 else float("inf")
    w = sys.stderr.write
    w("tokenizer         : " + ("tiktoken cl100k_base" if HAVE_TOK else "PROXY (~4c/tok, NOT representative)") + "\n")
    w(f"original tokens   : {orig}\n")
    w(f"compressed body   : {comp}\n")
    w(f"dictionary tokens : {legend_tok} (sent once)\n")
    w(f"total with dict   : {total}\n")
    w(f"tokens saved      : {orig - total} ({100 * (orig - total) / max(orig, 1):.1f}%)\n")
    if be != float("inf"):
        w(f"break-even turns  : {be:.2f}\n")
    w(f"usable symbols    : {len(usable_symbols())} of {len(SYMBOL_POOL)} pool entries\n")
    w("dictionary:\n")
    for line in legend:
        w("  " + line + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic phrase->symbol prompt compression.")
    ap.add_argument("--input", help="input file (default: stdin)")
    ap.add_argument("--output", help="output file (default: stdout)")
    ap.add_argument("--top", type=int, default=15, help="max mapping entries (capped by usable symbols)")
    ap.add_argument("--max-ngram", type=int, default=3, help="max English phrase length (words)")
    ap.add_argument("--cjk-min", type=int, default=2, help="min CJK phrase length (chars)")
    ap.add_argument("--cjk-max", type=int, default=6, help="max CJK phrase length (chars)")
    ap.add_argument("--no-legend", action="store_true", help="omit the dictionary header")
    ap.add_argument("--stats", action="store_true", help="print stats to stderr")
    ap.add_argument("--expand", action="store_true",
                    help="reverse mode: restore a compressed text (reads its own legend or --dictionary)")
    ap.add_argument("--dictionary", help="dictionary file for --expand (when input has no legend)")
    ap.add_argument("--allow-proxy", action="store_true",
                    help="allow running without tiktoken (counts are approximate and CJK gets NO mapping)")
    args = ap.parse_args(argv)

    if args.expand:
        text = _read_input(args.input)
        mapping = parse_legend(text)
        if not mapping and args.dictionary:
            mapping = load_dictionary(args.dictionary)
        if not mapping:
            sys.stderr.write("error: no dictionary found (input has no legend; pass --dictionary)\n")
            return 2
        _write_output(args.output, expand(strip_legend(text), mapping))
        if args.stats:
            sys.stderr.write(f"dictionary entries: {len(mapping)}\n")
        return 0

    text = _read_input(args.input)
    try:
        mapping, legend, legend_tok = build_mapping(
            text, args.top, args.max_ngram, args.cjk_min, args.cjk_max, args.allow_proxy)
    except TokenizerMissing as exc:
        sys.stderr.write("error: " + str(exc) + "\n")
        return 2

    if not HAVE_TOK:
        sys.stderr.write("warning: running with the ~4 chars/token proxy; numbers are estimates only\n")

    compressed = compress(text, mapping)
    legend_text = ""
    if legend and not args.no_legend:
        legend_text = "Dictionary (expand symbols before use):\n" + "\n".join(legend) + "\n\n"
    out = legend_text + compressed
    _write_output(args.output, out)

    if HAVE_TOK and args.top > len(usable_symbols()):
        sys.stderr.write(
            f"note: --top {args.top} exceeds the {len(usable_symbols())} symbols that cost 1 token; "
            f"dictionary capped at {len(usable_symbols())} entries\n")

    if args.stats:
        _emit_stats(text, mapping, legend, legend_tok, compressed, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
