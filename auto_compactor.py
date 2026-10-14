#!/usr/bin/env python3
"""
auto_compactor.py — 请求拦截层：每轮自动压缩上下文（方案 B）

把"高频词→单字符"的确定性词典压缩做成透明中间件。你照常调用
session.chat(user_msg)，它在底层自动完成"建字典→压缩历史→发送→存原文"，
对上层调用方完全透明；可选地把模型输出也压成符号、本地再解压（双端压缩）。

────────────────────────────────────────────────────────────────
用法 1：dry-run 演示（无需联网 / API key）
    python auto_compactor.py                      # 跑内置多轮演示
    python auto_compactor.py --rounds 8           # 指定轮数
    python auto_compactor.py --dict my_dict.txt   # 加载 Skill A 预置词典

用法 2：接入真实 OpenAI 兼容客户端
    from openai import OpenAI
    client = OpenAI()
    sess = AutoCompactSession(transport=client.chat.completions.create)
    sess.system = "你是一个助手"
    reply = sess.chat("用户的第一句话", model="gpt-4o")   # 自动压缩后发送

用法 3：复用 Skill A 建好的固定词典（推荐给稳定术语库）
    # Skill A 先产出一份 {sym} = {phrase} 图例，存成文件
    sess = AutoCompactSession(predefined_dict="my_dict.txt")
    # 或运行时加载：
    sess.load_dictionary("my_dict.txt")
    # 固定字典模式：每轮不再重建，术语表稳定、最省最稳
    # 若想让 B 在固定词典之外再补发现新词：allow_grow=True

核心思想：默认每轮用全量历史（原文）重建字典，压缩除"最近 N 条用户消息"以外
的所有历史，把图例拼进 system 消息只发一次。历史越滚越大，压缩省得越多。
有预置词典时，字典固定不变，跨轮稳定。
────────────────────────────────────────────────────────────────
"""

import re
import sys
import json
import argparse
from collections import Counter

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    def n_tokens(s: str) -> int:
        return len(ENC.encode(s))
    HAVE_TOK = True
except Exception:
    HAVE_TOK = False
    def n_tokens(s: str) -> int:          # crude proxy: ~4 chars per token
        return max(1, len(s) // 4)


# 候选符号池：只取经 tokenizer 校验为 1 token 的字符（很多 emoji 实际是多 token）
CANDIDATES = "§¶▪◆●★☆♠♣♥♦⬡⬢◼◻⬣▲▼◀▶♀♂☀☂☎☘♪♫☯☮✝✡⚡🔥❄⚙⚘⚛⚜⬧⬨❖✪✿"


# ───────────── 核心压缩（与 context-compactor 同源，自包含） ─────────────

def latin_words(text):
    return re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*", text)


def cjk_runs(text):
    return re.findall(r"[一-鿿]+", text)


def latin_ngrams(words, max_n):
    counts = Counter()
    n = len(words)
    for k in range(1, max_n + 1):
        for i in range(n - k + 1):
            counts[" ".join(words[i:i + k])] += 1
    return counts


def cjk_ngrams(run, min_l, max_l):
    counts = Counter()
    L = len(run)
    for k in range(min_l, max_l + 1):
        for i in range(L - k + 1):
            counts[run[i:i + k]] += 1
    return counts


def is_subphrase(cand, chosen):
    return cand in chosen


def cjk_overlap(a, b):
    """True if two CJK phrases share a >=4-char common substring (adjacent windows)."""
    if not (re.search(r"[一-鿿]", a) and re.search(r"[一-鿿]", b)):
        return False
    if len(a) < 4 or len(b) < 4:
        return a in b or b in a
    for i in range(len(a) - 3):
        if a[i:i + 4] in b:
            return True
    return False


def pick_symbols(n):
    out = []
    for ch in CANDIDATES:
        if n_tokens(ch) == 1:        # only 1-token symbols actually compress
            out.append(ch)
        if len(out) >= n:
            break
    return out


def compress(text, mapping):
    # longest phrases first so inner phrases are not double-replaced
    for phrase in sorted(mapping, key=len, reverse=True):
        sym = mapping[phrase]
        if re.search(r"[A-Za-z0-9]", phrase):
            rx = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
            text = re.sub(rx, sym, text)
        else:
            text = text.replace(phrase, sym)
    return text


def build_mapping(text, top, max_ngram, cjk_min, cjk_max):
    counts = Counter()
    counts.update(latin_ngrams(latin_words(text), max_ngram))
    for run in cjk_runs(text):
        counts.update(cjk_ngrams(run, cjk_min, cjk_max))
    raw = []
    for phrase, cnt in counts.items():
        tlen = n_tokens(phrase)
        if tlen < 2 or cnt < 2:          # 1-token or single-occurrence: no gain
            continue
        raw.append((cnt * (tlen - 1), cnt, tlen, phrase))   # saving = cnt*(tlen-1)
    raw.sort(reverse=True)                # biggest savers first
    chosen = []
    for saving, cnt, tlen, phrase in raw:
        skip = False
        for _, _, _, cp in chosen:
            if phrase in cp or cp in phrase or cjk_overlap(phrase, cp):
                skip = True               # covered by a longer entry, or adjacent window
                break
        if skip:
            continue
        if saving <= n_tokens("x = " + phrase):
            continue                      # cannot earn its own dictionary slot
        chosen.append((saving, cnt, tlen, phrase))
        if len(chosen) >= top:
            break
    syms = pick_symbols(len(chosen))
    mapping, legend = {}, []
    for (saving, cnt, tlen, phrase), sym in zip(chosen, syms):
        mapping[phrase] = sym
        legend.append(f"{sym} = {phrase}")
    return mapping, legend


def _looks_symbol(s):
    """Heuristic: True if `s` looks like a single compression symbol (not a phrase)."""
    s = str(s)
    return 0 < len(s) <= 3 and not re.search(r"[一-鿿A-Za-z0-9 ]", s)


def load_dictionary(path):
    """Load a pre-built dictionary produced by Skill A (context-compactor).

    Accepts two file formats:
      1. Legend text — one entry per line, ``SYM = PHRASE`` (exactly what
         context-compactor prints in its legend / --stats). Header lines
         ("Dictionary ...", "expand ...") and blank lines are ignored. The
         compressed body (if present) is harmlessly skipped.
      2. JSON (.json) — either {phrase: symbol} or {symbol: phrase}; the
         orientation is auto-detected from the keys.

    Returns a {phrase: symbol} mapping (same orientation as build_mapping()).
    """
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    mapping = {}
    if path.endswith(".json"):
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                keys = list(data)
                if keys and _looks_symbol(keys[0]):        # {sym: phrase} -> invert
                    for sym, phrase in data.items():
                        mapping[str(phrase)] = str(sym)
                else:                                       # {phrase: symbol}
                    for phrase, sym in data.items():
                        mapping[str(phrase)] = str(sym)
            return mapping
        except Exception:
            pass  # fall through to legend parsing
    # Legend text parsing: SYM = PHRASE
    for line in raw.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        low = line.lower()
        if low.startswith("dictionary") or low.startswith("expand"):
            continue
        sym, phrase = line.split("=", 1)
        sym, phrase = sym.strip(), phrase.strip()
        if not sym or not phrase:
            continue
        mapping[phrase] = sym
    return mapping


# ───────────── 请求拦截层 ─────────────

class DryRunTransport:
    """演示用传输层：不联网，记录发出的 payload 并返回假回复。"""
    def __init__(self):
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
    """
    请求拦截层。把任意"发送 messages → 返回回复"的传输函数包进来，
    每次 chat() 自动：建字典 → 压缩历史（仅保留最近 N 条用户消息原文）
    → 拼装 system+图例+压缩历史 → 调传输 → 存原文。对调用方透明。

    真实使用时：
        sess = AutoCompactSession(transport=openai_client.chat.completions.create)
        sess.chat("你好", model="gpt-4o")

    复用 Skill A 的固定词典（推荐给稳定术语库）：
        sess = AutoCompactSession(predefined_dict="dict.txt")   # 固定字典，每轮不重建
        # 或允许在固定词典之外再补发现新词：
        sess = AutoCompactSession(predefined_dict="dict.txt", allow_grow=True)
    """

    def __init__(self, transport=None, top=15, max_ngram=3, cjk_min=2, cjk_max=12,
                 keep_raw=1, verbose=False, predefined_dict=None, allow_grow=False):
        self.transport = transport or DryRunTransport()
        self.top = top
        self.max_ngram = max_ngram
        self.cjk_min = cjk_min
        self.cjk_max = cjk_max
        self.keep_raw = keep_raw          # 最近 N 条用户消息保持原文
        self.verbose = verbose
        self.allow_grow = allow_grow      # 预置词典之外，是否每轮补发现新词
        if isinstance(predefined_dict, str):
            predefined_dict = load_dictionary(predefined_dict)
        self.predefined_dict = predefined_dict   # {phrase: symbol} 或 None
        self.system = ""
        self.history = []                 # [{"role","content"}] 始终存原文

    def load_dictionary(self, path):
        """Load a pre-built dictionary from disk (Skill A output)."""
        self.predefined_dict = load_dictionary(path)
        return self.predefined_dict

    def _free_symbol(self, used):
        """Next 1-token candidate symbol not already in `used`, or None."""
        for ch in CANDIDATES:
            if n_tokens(ch) == 1 and ch not in used:
                return ch
        return None

    def _build_active_mapping(self):
        """The {phrase: symbol} mapping used for the outgoing payload.

        - No predefined dict  -> rebuild from full history every turn (default).
        - Predefined dict     -> use it as-is (stable term base, recommended).
        - Predefined + grow   -> seed with predefined, then add up to `top`
                                 newly-discovered high-frequency phrases.
        """
        if self.predefined_dict is None:
            return build_mapping(
                "\n".join(m["content"] for m in self.history
                          if m["role"] in ("user", "assistant")),
                self.top, self.max_ngram, self.cjk_min, self.cjk_max)
        base = dict(self.predefined_dict)
        if self.allow_grow:
            full = "\n".join(m["content"] for m in self.history
                             if m["role"] in ("user", "assistant"))
            grown, _ = build_mapping(full, self.top, self.max_ngram,
                                     self.cjk_min, self.cjk_max)
            used = set(base.values())
            for phrase, sym in grown.items():
                if phrase in base:
                    continue
                if any(phrase in p or p in phrase or cjk_overlap(phrase, p)
                       for p in base):
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
        out = []
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

    def chat(self, message, role="user", **kwargs):
        self.history.append({"role": role, "content": message})
        outgoing, mapping, legend = self._outgoing()
        if self.verbose:
            orig = n_tokens("\n".join(m["content"] for m in self.history))
            sent = n_tokens("\n".join(m["content"] for m in outgoing))
            saved = orig - sent
            pct = 100 * saved / max(orig, 1)
            print(f"  [intercept] turn={len(self.history)//2} "
                  f"dict={len(mapping)} history={orig}tok -> sent={sent}tok "
                  f"(省 {saved}tok, {pct:.1f}%)")
        reply = self.transport(messages=outgoing, **kwargs)
        text = reply.choices[0].message.content
        self.history.append({"role": "assistant", "content": text})
        return text

    def stats(self):
        out, mapping, legend = self._outgoing()
        orig = n_tokens("\n".join(m["content"] for m in self.history))
        sent = n_tokens("\n".join(m["content"] for m in out))
        return {"dict_size": len(mapping), "history_tokens": orig,
                "sent_tokens": sent, "saved": orig - sent,
                "saved_pct": 100 * (orig - sent) / max(orig, 1)}


# ───────────── 内置演示 ─────────────

def demo(rounds=6, dict_path=None, grow=False):
    sess = AutoCompactSession(verbose=True, allow_grow=grow)
    sess.system = "You are a helpful assistant."
    if dict_path:
        sess.load_dictionary(dict_path)
        print(f"[demo] 已加载预置词典: {dict_path} (固定字典模式, allow_grow={grow})")
    base = ("大型语言模型是一种模型。上下文窗口限制可见范围。"
            "训练数据决定模型能力。大型语言模型需要训练数据，"
            "上下文窗口决定神经网络能记住多少。")
    print("=== 方案 B：请求拦截层自动压缩演示（dry-run，无需 API key）===")
    print(f"tokenizer: {'tiktoken cl100k_base' if HAVE_TOK else 'proxy (~4c/tok)'}\n")
    for t in range(rounds):
        sess.chat(base + f"（这是第 {t+1} 轮提问）")
    s = sess.stats()
    print(f"\n最终：字典 {s['dict_size']} 条 | 原文历史 {s['history_tokens']}tok "
          f"-> 实际发送 {s['sent_tokens']}tok | 累计省 {s['saved']}tok ({s['saved_pct']:.1f}%)")
    print("\nsystem 消息里的图例（只发一次）：")
    print(sess._outgoing()[0][0]["content"][:400])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6, help="演示轮数")
    ap.add_argument("--dict", help="预置词典路径（Skill A 产出的 legend/JSON），固定字典模式")
    ap.add_argument("--grow", action="store_true", help="预置词典之外允许每轮补发现新词")
    args = ap.parse_args()
    demo(args.rounds, args.dict, args.grow)
