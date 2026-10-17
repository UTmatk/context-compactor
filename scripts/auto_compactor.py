#!/usr/bin/env python3
# Copyright (C) 2026 UTmatk
# SPDX-License-Identifier: GPL-3.0-or-later
"""
auto_compactor.py — 请求拦截层：每轮自动压缩上下文（方案 B）

把"高频词→单字符"的确定性词典压缩做成透明中间件。你照常调用
session.chat(user_msg)，它在底层自动完成"建字典→压缩历史→发送→存原文"，
对上层调用方透明。

────────────────────────────────────────────────────────────────
用法 1：dry-run 演示（无需联网 / API key）
    python scripts/auto_compactor.py                    # 跑内置多轮演示
    python scripts/auto_compactor.py --rounds 8
    python scripts/auto_compactor.py --dict my_dict.txt # 加载 Skill A 预置词典

用法 2：接入真实 OpenAI 兼容客户端
    from openai import OpenAI
    from auto_compactor import AutoCompactSession
    client = OpenAI()
    sess = AutoCompactSession(transport=client.chat.completions.create)
    sess.system = "你是一个助手"
    reply = sess.chat("用户的第一句话", model="gpt-4o")

用法 3：复用 Skill A 建好的固定词典（推荐给稳定术语库）
    sess = AutoCompactSession(predefined_dict="my_dict.txt")
    sess.load_dictionary("my_dict.txt")

依赖：tiktoken（精确计数）。缺失时 build_mapping 会抛 TokenizerMissing，
而不是静默产出一份 0 收益的空字典。
────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import json
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


# ─────────────────────────── 核心压缩（与 scripts/compact.py 同源） ───────────────
# NOTE: 这两个文件刻意保持自包含（Skill 分发时只拷一个目录），因此算法是复制的。
#       任何算法改动必须同步两处 —— tests/selftest.py 会校验两边输出一致。

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
    """True if two CJK phrases share a >=4-char substring (shifted windows)."""
    if not (re.search(r"[一-鿿]", a) and re.search(r"[一-鿿]", b)):
        return False
    if len(a) < 4 or len(b) < 4:
        return a in b or b in a
    for i in range(len(a) - 3):
        if a[i:i + 4] in b:
            return True
    return False


def overlaps(cand: str, chosen: str) -> bool:
    return cand in chosen or chosen in cand or cjk_overlap(cand, chosen)


def phrase_regex(phrase: str) -> re.Pattern:
    """Regex used to locate/replace a phrase (Latin gets word boundaries)."""
    if re.search(r"[A-Za-z0-9]", phrase):
        return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")
    return re.compile(re.escape(phrase))


def is_fragment(phrase: str, cnt: int, counts: Counter, text: str) -> bool:
    """True if `phrase` is only ever a slice of a longer *repeated* unit.

    Judge by "does the one-char/one-word extension occur just as often?" so
    phrases truncated by punctuation or a run boundary are never dropped.
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


def _span_conflicts(cand: str, chosen, span_of) -> bool:
    """True if `cand` overlaps, inside the source text, a already-chosen phrase."""
    for a, b in span_of(cand):
        for _, _, _, cp in chosen:
            for x, y in span_of(cp):
                if a < y and x < b:
                    return True
    return False


def build_mapping(text: str, top: int = 15, max_ngram: int = 3,
                  cjk_min: int = 2, cjk_max: int = 12,
                  allow_proxy: bool = False):
    """Return (mapping {phrase: symbol}, legend lines)."""
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

    limit = min(top, len(usable_symbols()))

    # 位置级重叠守卫：光看"短语是不是另一个短语的子串"不够。选中「训练数据」后，
    # 文本里剩下的文字会拼出「练数据决定」这种跨边界的滑动窗碎片——它既不是子串、
    # 也不共享 4 字窗口，但和已选短语在原文里重叠。这里直接比对出现位置跨度。
    spans: dict[str, list[tuple[int, int]]] = {}

    def span_of(phrase: str) -> list[tuple[int, int]]:
        if phrase not in spans:
            spans[phrase] = [(m.start(), m.end())
                             for m in phrase_regex(phrase).finditer(text)]
        return spans[phrase]

    chosen = []
    for saving, cnt, tlen, phrase in raw:
        if any(overlaps(phrase, cp) for _, _, _, cp in chosen):
            continue                     # covered by a longer entry, or shifted window
        if saving <= n_tokens("§ = " + phrase):
            continue                     # cannot pay for its own dictionary slot
        if is_fragment(phrase, cnt, counts, text):
            continue                     # only ever a slice of a longer repeated unit
        if chosen and _span_conflicts(phrase, chosen, span_of):
            continue                     # shares text positions with a chosen phrase
        chosen.append((saving, cnt, tlen, phrase))
        if len(chosen) >= limit:
            break

    mapping: dict[str, str] = {}
    legend: list[str] = []
    for (saving, cnt, tlen, phrase), sym in zip(chosen, pick_symbols(len(chosen))):
        mapping[phrase] = sym
        legend.append(f"{sym} = {phrase}")
    return mapping, legend


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


# ─────────────────────────── 词典文件解析 ───────────────────────────

LEGEND_RE = re.compile(r"^(\S{1,4})\s*=\s*(.+)$")


def looks_like_symbol(s: str) -> bool:
    """A legend symbol: short, not alphanumeric/CJK, not an '=' or space."""
    return bool(s) and len(s) <= 4 and not re.search(r"[A-Za-z0-9_ =一-鿿]", s)


def parse_legend(text: str) -> dict[str, str]:
    """Parse a legend block into {phrase: symbol}.

    Stops at the first blank line after the block starts, so the compressed body
    that follows a legend (``...\\n\\n train = fit``) is never read as entries.
    """
    mapping: dict[str, str] = {}
    started = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if started and mapping:
                break                     # end of legend block
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


def load_dictionary(path: str) -> dict[str, str]:
    """Load a dictionary produced by Skill A (legend text, packed file, or JSON).

    Legend text: one ``SYM = PHRASE`` per line; a ``Dictionary ...`` header and
    blank lines are ignored, and anything after the legend block (the compressed
    body) is skipped. JSON: ``{phrase: symbol}`` or ``{symbol: phrase}``, the
    orientation is auto-detected from the first key.

    Returns a ``{phrase: symbol}`` mapping (same orientation as build_mapping()).
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    if path.lower().endswith(".json"):
        try:
            data = json.loads(raw)
        except Exception:
            data = None
        if isinstance(data, dict) and data:
            keys = list(data)
            if looks_like_symbol(str(keys[0])):        # {sym: phrase} -> invert
                return {str(p): str(s) for s, p in data.items()}
            return {str(p): str(s) for p, s in data.items()}
    return parse_legend(raw)


def audit_dictionary(mapping: dict[str, str]) -> list[str]:
    """Human-readable warnings: symbols that are not 1 token, duplicates, etc."""
    warnings: list[str] = []
    for phrase, sym in mapping.items():
        if not is_one_token_symbol(sym):
            warnings.append(f"symbol {sym!r} costs {n_tokens(sym)} tokens (>1) for phrase {phrase!r}")
    syms = list(mapping.values())
    dup = {s for s in syms if syms.count(s) > 1}
    for s in sorted(dup):
        warnings.append(f"symbol {s!r} is assigned to more than one phrase")
    return warnings


# ─────────────────────────── 请求拦截层 ───────────────────────────

class DryRunTransport:
    """演示用传输层：不联网，记录发出的 payload 并返回假回复。"""

    def __init__(self) -> None:
        self.last_payload = None
        self.turn = 0

    def __call__(self, messages, **kwargs):
        self.last_payload = messages
        self.turn += 1

        class _Resp:
            class _Choice:
                class _Msg:
                    content = f"[dry-run reply #{self.turn}] 已收到并理解上下文。"

                message = _Msg()

            choices = [_Choice()]

        return _Resp()


class AutoCompactSession:
    """请求拦截层：包住任意 "发送 messages → 返回回复" 的传输函数。

    每次 chat() 自动：建字典 → 压缩历史（最近 keep_raw 条用户消息保持原文）
    → 拼装 system+图例+压缩历史 → 调传输 → 存原文。对调用方透明。

    失败语义：传输函数抛异常时，本轮用户消息会从 history 里回滚，不留下
    孤儿轮次（下一次 chat() 不会看到一个没有回复的半截对话）。
    """

    def __init__(self, transport=None, top: int = 15, max_ngram: int = 3,
                 cjk_min: int = 2, cjk_max: int = 12, keep_raw: int = 1,
                 verbose: bool = False, predefined_dict=None, allow_grow: bool = False,
                 allow_proxy: bool = False):
        self.transport = transport or DryRunTransport()
        self.top = top
        self.max_ngram = max_ngram
        self.cjk_min = cjk_min
        self.cjk_max = cjk_max
        self.keep_raw = keep_raw          # 最近 N 条用户消息保持原文
        self.verbose = verbose
        self.allow_grow = allow_grow      # 预置词典之外，是否每轮补发现新词
        self.allow_proxy = allow_proxy
        if isinstance(predefined_dict, str):
            predefined_dict = load_dictionary(predefined_dict)
        self.predefined_dict = predefined_dict   # {phrase: symbol} or None
        self.system = ""
        self.history: list[dict[str, str]] = []  # [{"role","content"}] 始终存原文
        self.warnings = audit_dictionary(self.predefined_dict or {})

    def load_dictionary(self, path: str) -> dict[str, str]:
        """Load a pre-built dictionary from disk (Skill A output)."""
        self.predefined_dict = load_dictionary(path)
        self.warnings = audit_dictionary(self.predefined_dict)
        return self.predefined_dict

    def _free_symbol(self, used: set[str]) -> str | None:
        for ch in usable_symbols():
            if ch not in used:
                return ch
        return None

    def _history_text(self) -> str:
        return "\n".join(m["content"] for m in self.history
                         if m["role"] in ("user", "assistant"))

    def _build_active_mapping(self):
        """The {phrase: symbol} mapping used for the outgoing payload.

        - No predefined dict  -> rebuild from full history every turn (default).
        - Predefined dict     -> use it as-is (stable term base, recommended).
        - Predefined + grow   -> seed with predefined, then add up to `top`
                                 newly-discovered high-frequency phrases.
        """
        if self.predefined_dict is None:
            return build_mapping(self._history_text(), self.top, self.max_ngram,
                                 self.cjk_min, self.cjk_max, self.allow_proxy)

        base = dict(self.predefined_dict)
        if self.allow_grow:
            grown, _ = build_mapping(self._history_text(), self.top, self.max_ngram,
                                     self.cjk_min, self.cjk_max, self.allow_proxy)
            used = set(base.values())
            for phrase, sym in grown.items():
                if phrase in base:
                    continue
                if any(phrase in p or p in phrase or cjk_overlap(phrase, p) for p in base):
                    continue                # overlaps an existing entry
                if sym in used:
                    sym = self._free_symbol(used)
                    if sym is None:
                        break
                base[phrase] = sym
                used.add(sym)
                if len(base) >= self.top:
                    break
        legend = [f"{sym} = {phrase}" for phrase, sym in base.items()]
        return base, legend

    def _outgoing(self):
        mapping, legend = self._build_active_mapping()
        out: list[dict[str, str]] = []
        sys_text = self.system or ""
        if legend:
            sys_text = (sys_text + "\n\n" if sys_text else "") + \
                       "Dictionary (expand symbols before use):\n" + "\n".join(legend)
        if sys_text:
            out.append({"role": "system", "content": sys_text})
        n = len(self.history)
        for i, m in enumerate(self.history):
            content = m["content"]
            # 最近 keep_raw 条用户消息保持原文，让模型在完整文本上推理新问题
            if not (m["role"] == "user" and i >= n - self.keep_raw):
                content = compress(content, mapping)
            out.append({"role": m["role"], "content": content})
        return out, mapping, legend

    def chat(self, message: str, role: str = "user", **kwargs):
        self.history.append({"role": role, "content": message})
        try:
            outgoing, mapping, legend = self._outgoing()
            if self.verbose:
                orig = n_tokens("\n".join(m["content"] for m in self.history))
                sent = n_tokens("\n".join(m["content"] for m in outgoing))
                saved = orig - sent
                pct = 100 * saved / max(orig, 1)
                print(f"  [intercept] turn={len(self.history) // 2} "
                      f"dict={len(mapping)} history={orig}tok -> sent={sent}tok "
                      f"(省 {saved}tok, {pct:.1f}%)")
            reply = self.transport(messages=outgoing, **kwargs)
            text = reply.choices[0].message.content
        except BaseException:
            self.history.pop()            # 回滚：不留孤儿轮次
            raise
        self.history.append({"role": "assistant", "content": text})
        return text

    def stats(self) -> dict:
        out, mapping, legend = self._outgoing()
        orig = n_tokens("\n".join(m["content"] for m in self.history))
        sent = n_tokens("\n".join(m["content"] for m in out))
        return {"dict_size": len(mapping), "history_tokens": orig,
                "sent_tokens": sent, "saved": orig - sent,
                "saved_pct": 100 * (orig - sent) / max(orig, 1)}


# ─────────────────────────── 内置演示 ───────────────────────────

DEMO_BASE = ("大型语言模型是一种模型。上下文窗口限制可见范围。"
             "训练数据决定模型能力。大型语言模型需要训练数据，"
             "上下文窗口决定神经网络能记住多少。")


def demo(rounds: int = 6, dict_path: str | None = None, grow: bool = False,
         allow_proxy: bool = False) -> int:
    sess = AutoCompactSession(verbose=True, allow_grow=grow, allow_proxy=allow_proxy)
    sess.system = "You are a helpful assistant."
    if dict_path:
        sess.load_dictionary(dict_path)
        print(f"[demo] 已加载预置词典: {dict_path} (固定字典模式, allow_grow={grow})")
        for w in sess.warnings:
            print("[demo] 词典告警: " + w)
    print("=== 方案 B：请求拦截层自动压缩演示（dry-run，无需 API key）===")
    print(f"tokenizer: {'tiktoken cl100k_base' if HAVE_TOK else 'proxy (~4c/tok)'}\n")
    try:
        for t in range(rounds):
            sess.chat(DEMO_BASE + f"（这是第 {t + 1} 轮提问）")
    except TokenizerMissing as exc:
        sys.stderr.write("error: " + str(exc) + "\n")
        return 2
    s = sess.stats()
    print(f"\n最终：字典 {s['dict_size']} 条 | 原文历史 {s['history_tokens']}tok "
          f"-> 实际发送 {s['sent_tokens']}tok | 累计省 {s['saved']}tok ({s['saved_pct']:.1f}%)")
    print("\nsystem 消息里的图例（每次请求随附一份）：")
    print(sess._outgoing()[0][0]["content"][:400])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transparent per-turn context compression.")
    ap.add_argument("--rounds", type=int, default=6, help="演示轮数")
    ap.add_argument("--dict", help="预置词典路径（Skill A 产出的 legend/JSON），固定字典模式")
    ap.add_argument("--grow", action="store_true", help="预置词典之外允许每轮补发现新词")
    ap.add_argument("--allow-proxy", action="store_true", help="无 tiktoken 时也跑（计数不准确）")
    args = ap.parse_args(argv)
    return demo(args.rounds, args.dict, args.grow, args.allow_proxy)


if __name__ == "__main__":
    raise SystemExit(main())
