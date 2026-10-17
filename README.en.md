# context-compactor

Replace high-frequency phrases in long documents and conversation history with **single-token symbols** so you send fewer tokens to the model. Deterministic and reversible — no small model in the loop.

```text
original : A large language model is a language model built on a neural network. The context window ...
legend   : § = large language model   ¶ = context window   ● = training data
packed   : A § is a language model built on a neural network. The ¶ ...
```

[简体中文](README.md) · **English**

---

## What it does

| Form | Script | Use it for |
|---|---|---|
| **A — manual** | `scripts/compact.py` | Compress a long document once, or derive a fixed dictionary from a corpus |
| **B — transparent interceptor** | `scripts/auto_compactor.py` | Wrap your `chat()` and compress history on every turn, invisible to callers |

Both share the same engine: count frequent n-grams → five filters to keep only phrases worth encoding → assign single-token symbols → longest-first replacement. The recommended setup is **build the dictionary with A, apply it with B** — not chaining both over the same text.

## Quick start

Requires Python ≥ 3.10 and `tiktoken` (**hard dependency**).

```bash
pip install -r requirements.txt

# A: compress a file and print stats
python scripts/compact.py --input long.txt --output packed.txt --stats

# A: restore (reads the legend embedded in packed.txt)
python scripts/compact.py --expand --input packed.txt --output restored.txt

# B: offline demo, no API key needed
python scripts/auto_compactor.py --rounds 6

# self-test: 26 regression checks
python tests/selftest.py
```

You can also grab the packaged skills from [Releases](https://github.com/UTmatk/context-compactor/releases) (`context-compactor.zip` / `context-auto-compactor.zip`) and unzip them into `.dsh/skills/` or `~/.workbuddy/skills/` to use them as triggerable Skills.

---

## Form A: compress a piece of text

### Run it

```bash
python scripts/compact.py --input long.txt --output packed.txt --stats
```

Real output (172-character Chinese sample):

```
original tokens   : 186
compressed body   : 95
dictionary tokens : 36 (sent once)
total with dict   : 143
tokens saved      : 43 (23.1%)
break-even turns  : 0.40
usable symbols    : 38 of 38 pool entries
dictionary:
  § = 大型语言模型
  ¶ = 训练数据
  ● = 上下文窗口
  ★ = 神经网络
  ☆ = 决定
```

`packed.txt` looks like this:

```
Dictionary (expand symbols before use):
§ = 大型语言模型
¶ = 训练数据
...

§是一种基于★的语言模型。●决定§一次能处理多少文本。……
```

**Reading the numbers**: `break-even turns 0.40` means the legend pays for itself once the same terms recur 0.4 more times. Below 1 turn it is worth using; above ~5 the text simply is not repetitive enough.

### Options

| Flag | Default | When to change it |
|---|---|---|
| `--top` | 15 | Want more entries (**hard cap 38** — only that many symbols cost one token; the tool warns when you exceed it) |
| `--cjk-max` | 6 | Raise to 8–12 when long Chinese terms get chopped |
| `--max-ngram` | 3 | Raise for longer English phrases |
| `--cjk-min` | 2 | Rarely touched |
| `--no-legend` | off | You manage the legend yourself |
| `--stats` | off | Keep it on when you need to judge whether compression pays off |
| `--expand` | off | Reverse mode: restore text |
| `--dictionary` | — | Dictionary file for `--expand` when the input has no legend |
| `--allow-proxy` | off | Run without tiktoken (numbers become meaningless) |

---

## Form B: automatic per-turn compression

### See it work (no API key)

```bash
python scripts/auto_compactor.py --rounds 6
```

```
  [intercept] turn=3 dict=5 history=390tok -> sent=346tok (省 44tok, 11.3%)
  ...
最终：字典 7 条 | 原文历史 612tok -> 实际发送 407tok | 累计省 140tok (33.5%)
```

The first turns are negative because the legend is not amortized yet; after that it turns positive and keeps improving as history grows.

### Wire it to a real model

```python
from openai import OpenAI
from auto_compactor import AutoCompactSession      # root shim, or the same module under scripts/

client = OpenAI()
sess = AutoCompactSession(
    transport=client.chat.completions.create,
    keep_raw=1,          # keep the last user message raw — never compress the current question
)
sess.system = "You are a project assistant."

while True:
    msg = input("you> ")
    print("bot>", sess.chat(msg, model="gpt-4o"))
    print(sess.stats())    # {'dict_size': 7, 'saved_pct': 33.5, ...}
```

### Options

| Argument | Default | Meaning |
|---|---|---|
| `transport` | `DryRunTransport` | Your send function; use `client.chat.completions.create` in production |
| `keep_raw` | 1 | How many trailing user messages stay uncompressed |
| `top` | 15 | Max dictionary entries |
| `max_ngram` | 3 | Max English phrase length in words |
| `cjk_min` / `cjk_max` | 2 / 12 | Chinese phrase length range |
| `predefined_dict` | None | Fixed dictionary (dict or path); non-None switches to fixed-dictionary mode |
| `allow_grow` | False | Allow discovering extra phrases on top of a fixed dictionary |
| `verbose` | False | Print per-turn compression details |
| `allow_proxy` | False | Run without tiktoken (counts are unreliable) |

Also worth knowing:

- `sess.stats()` → `{"dict_size", "history_tokens", "sent_tokens", "saved", "saved_pct"}`.
- **Rollback on failure**: if the transport raises, the current user message is popped from `history` — no orphan turn is left behind.
- `sess.warnings`: sanity report for a hand-written dictionary (symbols that are not one token, duplicate assignments).

---

## Recommended: build with A, apply with B

The most effective and most stable setup, especially for jargon-heavy projects.

```bash
# Step 1: derive a fixed dictionary from a domain corpus
python scripts/compact.py --input project_corpus.txt --output project_dict.txt --stats
```

```python
# Step 2: let the interceptor load it — the dictionary never changes again
sess = AutoCompactSession(
    transport=client.chat.completions.create,
    predefined_dict="project_dict.txt",   # ← the important part
)
```

The output of `compact.py` (legend + compressed body) works directly as a dictionary: B parses only the legend block and ignores everything after the first blank line.

Three modes measured over the same 4 Chinese turns:

| Mode | Dictionary | Cumulative saving |
|---|---|---|
| Rebuild every turn (default) | 7 entries, changing | 33.5% (6 turns) |
| Fixed dictionary | always 4 entries | 18.6% |
| Fixed + `allow_grow` | 4 → 8 entries | 25.0% |

**Why fixed usually wins in practice**: a stable legend means prompt caching can hit. In the default rebuild mode the system message changes every turn, which invalidates the cache and can eat the savings on your bill.

---

## Dictionary file format

One entry per line, `SYMBOL = PHRASE`. The symbol **must** cost exactly one token:

```
Dictionary (expand symbols before use):
§ = 大型语言模型
¶ = 上下文窗口
← = 训练数据
● = 神经网络
```

- JSON works too, in either direction: `{"大型语言模型": "§"}` or `{"§": "大型语言模型"}`.
- A `Dictionary ...` / `expand ...` header, blank lines and `#` comments are ignored.
- Only the **legend block** is parsed; anything after it (e.g. the compressed body inside a packed file) is skipped and never mistaken for entries.
- ⚠️ Avoid `◆ ▲ ⚡ 🔥 ◼` — those cost **2 tokens** in cl100k_base, so they make compression a net loss.
- **38 symbols known to be one token each**:
  `§ ¶ ● ★ ☆ ♥ ♀ ♪ ← ↑ → ↓ ─ ━ │ █ ░ † ¬ ± α β γ δ ε η θ κ λ μ π ρ σ τ υ φ χ ω`
- Ready-made samples: `test_dict.txt` / `test_dict.json`.

---

## When not to use it

- **Short texts, one-off questions**: the legend cannot be amortized; `--stats` will report break-even above 1 turn.
- **Creative writing**: symbol substitution degrades style.
- **Low-frequency content**: it cannot shrink "rare but bulky" passages — use summarization or RAG for those.
- **Long text with little repetition**: check break-even before committing.

Replacement is **literal** and has no word-boundary awareness for Chinese; when applying a dictionary to out-of-domain text, validate with `--expand` first.

---

## How phrases are chosen

Candidates are ranked by `occurrences × (original tokens − 1)` and then filtered five times:

1. **Substring** — skip phrases already covered by a longer chosen entry;
2. **4-character window** — skip Chinese phrases sharing a ≥4-char window with a chosen entry;
3. **Maximality** — if every occurrence is flanked by the same neighbour and the longer string occurs just as often, the phrase is only a slice of a longer repeated unit (`言模型是一种模型` ← `语言模型是一种模型`), so skip it;
4. **Source span** — skip any candidate whose occurrences overlap a chosen phrase in the original text (this kills boundary-straddling fragments like `练数据决定`);
5. **Cost** — a phrase must earn back the token cost of its own legend line.

English additionally rejects phrases that straddle sentence boundaries (second word capitalized, e.g. `data The context`). The symbol pool is validated at startup with `n_tokens(sym) == 1`.

Without these guards the picker happily emits half-words like `言模型是一种模型` — they look like savings but are not reusable.

---

## Benchmarks (tiktoken cl100k_base)

| Scenario | Command | Result |
|---|---|---|
| English sample, 696 tokens | `context_compactor_demo.py` | → **524**, saves **24.7%**, break-even 0.13 turns |
| Chinese sample, 186 tokens | `compact.py --stats` | → **143**, saves **23.1%**, break-even 0.40 turns |
| 6-turn chat (default rebuild) | `--rounds 6` | 612 → 407 tok, saves **33.5%** |
| 4 turns with fixed dictionary | `predefined_dict=test_dict.txt` | 408 → 332 tok, saves **18.6%** |
| 4 turns with growth | `allow_grow=True` | 4 → 8 entries, saves **25.0%** |

The ceiling depends on how much of the text is made of repeated phrases. Corpora dense with terminology, proper nouns or identifiers commonly land in the 30–50% range.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TokenizerMissing` | tiktoken not installed | `pip install -r requirements.txt` |
| 0% saved, empty dictionary | ran with `--allow-proxy`, or no tiktoken | Same as above |
| `◆ ▲` show up in the legend | hand-written dictionary uses 2-token symbols | Use symbols from this README; check `sess.warnings` |
| Model answers got worse | the current question was compressed | Raise `keep_raw`; only compress history |
| Tokens saved but the bill did not drop | default mode changes the legend every turn, breaking prompt caching | Switch to a fixed dictionary |
| Slow on multi-MB corpora | every candidate is regex-scanned | Compress in chunks (~0.07 s per 18K Chinese chars) |

---

## Repository layout

```
scripts/compact.py              Skill A — canonical implementation (compress + --expand)
scripts/auto_compactor.py       Skill B — canonical implementation (interceptor)
auto_compactor.py               Compatibility shim re-exporting scripts/auto_compactor.py
context-compactor/              Skill A docs (SKILL.md, packaged into the zip)
context-auto-compactor/         Skill B docs (SKILL.md, packaged into the zip)
context-compactor.zip           Distributable skill package (SKILL.md + scripts/ + LICENSE)
context-auto-compactor.zip      Same for B
tests/selftest.py               26 regression checks
build_packages.py               Deterministic zip rebuild (fixed timestamps)
context_compactor_demo.py       Teaching demo (calls the canonical implementation)
test_dict.txt / .json           Dictionary format samples
思路.md                          Design notes, algorithm, risks and changelog (Chinese)
```

## License

GPL-3.0-or-later — see [LICENSE.txt](LICENSE.txt).
