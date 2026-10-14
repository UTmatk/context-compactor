#!/usr/bin/env python3
"""Prompt dictionary compaction demo (v2: non-overlapping greedy selection).

Maps high-frequency phrases to single-token symbols to shrink context.
Key fix vs v1: a phrase that is a sub-phrase of an already-chosen longer
phrase is skipped, so we never waste a dictionary slot on text that the
longer entry already compresses.
"""
import re

try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    def n_tokens(s: str) -> int:
        return len(ENC.encode(s))
    HAVE_TOK = True
except Exception:
    HAVE_TOK = False
    def n_tokens(s: str) -> int:
        return max(1, len(s) // 4)


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

CANDIDATES = list("§¶▪◆●★☆♠♣♥♦⬡⬢◼◻⬣▲▼◀▶♀♂☀☂☎☘♪♫☯☮✝✡⚡🔥❄⚙⚘⚛⚜⬧⬨❖✪✿")


def words_of(text):
    return re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*|[一-鿿]|[^\s\w]", text)


def ngram_counts(wordlist, max_n=3):
    counts = {}
    n = len(wordlist)
    for k in range(1, max_n + 1):
        for i in range(n - k + 1):
            phrase = " ".join(wordlist[i:i + k])
            counts[phrase] = counts.get(phrase, 0) + 1
    return counts


def is_subphrase(candidate, chosen_phrase):
    # candidate appears as a contiguous sub-phrase inside chosen_phrase
    if candidate == chosen_phrase:
        return True
    p = " " + chosen_phrase + " "
    c = " " + candidate + " "
    return c in p


def pick_symbols(n):
    out = []
    for ch in CANDIDATES:
        if n_tokens(ch) == 1:        # only 1-token symbols compress anything
            out.append(ch)
        if len(out) >= n:
            break
    return out


def compress(text, mapping):
    for phrase in sorted(mapping, key=len, reverse=True):
        sym = mapping[phrase]
        rx = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        text = re.sub(rx, sym, text)
    return text


def main():
    text = SAMPLE
    words = words_of(text)
    counts = ngram_counts(words, max_n=3)

    raw = []
    for phrase, cnt in counts.items():
        tlen = n_tokens(phrase)
        if tlen < 2 or cnt < 2:
            continue
        saving = cnt * (tlen - 1)
        raw.append((saving, cnt, tlen, phrase))

    raw.sort(reverse=True)

    # non-overlapping greedy: skip any phrase already covered by a chosen longer one
    chosen = []
    for saving, cnt, tlen, phrase in raw:
        if any(is_subphrase(phrase, cp) for _, _, _, cp in chosen):
            continue
        legend_cost = n_tokens("x = " + phrase)   # approx cost of one dict entry
        if saving <= legend_cost:                 # must earn its own slot
            continue
        chosen.append((saving, cnt, tlen, phrase))

    symbols = pick_symbols(len(chosen))
    mapping = {}
    legend_lines = []
    legend_tok = 0
    for (saving, cnt, tlen, phrase), sym in zip(chosen, symbols):
        mapping[phrase] = sym
        line = f"{sym} = {phrase}"
        legend_lines.append(line)
        legend_tok += n_tokens(line)

    legend = "Dictionary (expand symbols before use):\n" + "\n".join(legend_lines) + "\n\n"
    compressed = compress(text, mapping)

    orig_tok = n_tokens(text)
    comp_only_tok = n_tokens(compressed)
    total_tok = n_tokens(legend + compressed)
    per_turn = orig_tok - comp_only_tok
    break_even = (legend_tok / per_turn) if per_turn > 0 else float("inf")

    print("=" * 64)
    print("PROMPT DICTIONARY COMPACTION DEMO  (v2, non-overlapping)")
    print("tokenizer :", "tiktoken cl100k_base" if HAVE_TOK else "proxy (~4 chars/tok)")
    print("=" * 64)
    print(f"original tokens            : {orig_tok}")
    print(f"compressed body tokens     : {comp_only_tok}")
    print(f"dictionary (legend) tokens: {legend_tok}  (sent ONCE)")
    print(f"total with dictionary      : {total_tok}")
    print(f"tokens saved               : {orig_tok - total_tok}  ({100*(orig_tok-total_tok)/orig_tok:.1f}%)")
    if break_even != float("inf"):
        print(f"break-even after           : {break_even:.2f} repeated turns")
    print("-" * 64)
    print("DICTIONARY:")
    for l in legend_lines:
        print("  " + l)
    print("-" * 64)
    print("COMPRESSED (head):")
    print(compressed[:460])


if __name__ == "__main__":
    main()
