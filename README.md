# context-compactor

把长文本 / 对话历史里高频重复的短语替换成**单 token 符号**，在发给模型前降低 token 占用。确定性、可逆，不依赖小模型。

```text
原文    : 大型语言模型是一种基于神经网络的语言模型。上下文窗口决定……
词典    : § = 大型语言模型   ¶ = 上下文窗口   ● = 训练数据
压缩后  : §是一种基于●的语言模型。¶决定……
```

**简体中文** · [English](README.en.md)

---

## 它能做什么

| 形态 | 脚本 | 用法 |
|---|---|---|
| **A 手动压缩** | `scripts/compact.py` | 一次性压缩长文档 / 在语料上产出一份固定词典 |
| **B 自动拦截层** | `scripts/auto_compactor.py` | 包住你的 `chat()`，每轮自动压缩历史，对调用方透明 |

两者算法同源：统计高频 n-gram → 五道筛挑出真正值得编码的短语 → 分配单 token 符号 → 最长优先替换。**推荐组合是 A 建库、B 套用**（见下文），不是把两者串在同一段文本上压两次。

## 快速开始

需要 Python ≥ 3.10 与 `tiktoken`（**硬依赖**）。

```bash
pip install -r requirements.txt

# A：压缩一个文件并看统计
python scripts/compact.py --input long.txt --output packed.txt --stats

# A：还原（读 packed.txt 自带的图例）
python scripts/compact.py --expand --input packed.txt --output restored.txt

# B：离线演示，不需要 API key
python scripts/auto_compactor.py --rounds 6

# 自检：26 项回归
python tests/selftest.py
```

也可以从 [Releases](https://github.com/UTmatk/context-compactor/releases) 直接下载技能包（`context-compactor.zip` / `context-auto-compactor.zip`），解压到 `.dsh/skills/` 或 `~/.workbuddy/skills/` 即为可触发的 Skill。

---

## 场景 A：手动压缩一段文本

### 跑一次

```bash
python scripts/compact.py --input long.txt --output packed.txt --stats
```

真实输出（172 字中文样本）：

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

`packed.txt` 长这样：

```
Dictionary (expand symbols before use):
§ = 大型语言模型
¶ = 训练数据
...

§是一种基于★的语言模型。●决定§一次能处理多少文本。……
```

**怎么读这几行**：`break-even turns 0.40` = 同一批术语再出现 0.4 次就回本。小于 1 就值得用；5 轮以上说明这段文本重复度不够，别压。

### 参数

| 参数 | 默认 | 什么时候调 |
|---|---|---|
| `--top` | 15 | 想要更多词条（**上限 38**，再多没有单 token 符号可用，会提示） |
| `--cjk-max` | 6 | 中文长术语被切碎时调大（8~12） |
| `--max-ngram` | 3 | 英文词组更长时调大 |
| `--cjk-min` | 2 | 一般不用动 |
| `--no-legend` | 关 | 你自己管映射表，不想每次带表头 |
| `--stats` | 关 | 想判断"值不值"就一定要开 |
| `--expand` | 关 | 反向模式：还原 |
| `--dictionary` | — | `--expand` 时输入里没有图例，用它指定 |
| `--allow-proxy` | 关 | 无 tiktoken 时也跑（数字不可信） |

---

## 场景 B：多轮对话自动压缩

### 先看效果（不用 API key）

```bash
python scripts/auto_compactor.py --rounds 6
```

```
  [intercept] turn=3 dict=5 history=390tok -> sent=346tok (省 44tok, 11.3%)
  ...
最终：字典 7 条 | 原文历史 612tok -> 实际发送 407tok | 累计省 140tok (33.5%)
```

前几轮图例没摊薄为负属正常，之后转正并随历史增长。

### 接真实模型

```python
from openai import OpenAI
from auto_compactor import AutoCompactSession      # 根目录垫片，或 scripts/ 下同模块

client = OpenAI()
sess = AutoCompactSession(
    transport=client.chat.completions.create,
    keep_raw=1,          # 最近 1 条用户消息保持原文，别压当前问题
)
sess.system = "你是本项目的助手"

while True:
    msg = input("你> ")
    print("助手>", sess.chat(msg, model="gpt-4o"))
    print(sess.stats())    # {'dict_size': 7, 'saved_pct': 33.5, ...}
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `transport` | `DryRunTransport` | 发送函数；真实场景传 `client.chat.completions.create` |
| `keep_raw` | 1 | 最近几条用户消息保持原文 |
| `top` | 15 | 字典条数上限 |
| `max_ngram` | 3 | 英文词组最大词数 |
| `cjk_min` / `cjk_max` | 2 / 12 | 中文词串长度区间 |
| `predefined_dict` | None | 预置词典（dict 或路径字符串），非空即进入固定字典模式 |
| `allow_grow` | False | 固定词典之外是否每轮补发现新词 |
| `verbose` | False | 逐轮打印压缩明细 |
| `allow_proxy` | False | 无 tiktoken 时也跑（计数不可信） |

其它要点：

- `sess.stats()` → `{"dict_size", "history_tokens", "sent_tokens", "saved", "saved_pct"}`。
- **失败回滚**：传输函数抛异常时本轮用户消息会从 `history` 弹出，不留没有回复的半截轮次。
- `sess.warnings`：加载手写词典后的体检结果（符号是否真 1 token、是否重复分配）。

---

## 推荐用法：A 建库 → B 套用

最省最稳，尤其适合满是行话、专有名词的项目。

```bash
# 第 1 步：拿领域语料建一份固定词典
python scripts/compact.py --input project_corpus.txt --output project_dict.txt --stats
```

```python
# 第 2 步：让拦截层加载它，字典从此不变
sess = AutoCompactSession(
    transport=client.chat.completions.create,
    predefined_dict="project_dict.txt",   # ← 关键
)
```

`compact.py` 的输出文件（含图例 + 压缩正文）**可以直接**当词典用——B 只解析图例段，第一个空行之后的正文整体忽略。

三种模式对比（实测，4 轮同一段中文）：

| 模式 | 字典 | 累计省 |
|---|---|---|
| 每轮重建（默认） | 7 条，逐轮变化 | 33.5%（6 轮） |
| 固定词典 | 恒 4 条 | 18.6% |
| 固定 + `allow_grow` | 4 → 8 条 | 25.0% |

**为什么固定词典反而更值得用**：图例跨轮恒定 → prompt caching 能命中。默认模式图例每轮都变，缓存全废，账单上可能把省下的又赔回去。

---

## 词典文件怎么写

一行一条 `符号 = 短语`，符号**必须 1 token**：

```
Dictionary (expand symbols before use):
§ = 大型语言模型
¶ = 上下文窗口
← = 训练数据
● = 神经网络
```

- 存 `.json` 也行，`{"大型语言模型": "§"}` 或反过来的方向都能自动识别。
- 表头 `Dictionary ...` / `expand ...`、空行、`#` 注释会被忽略。
- 只解析**图例块**：图例之后的内容（比如 packed 文件里的压缩正文）整体跳过，不会被误当成词条。
- ⚠️ 别用 `◆ ▲ ⚡ 🔥 ◼` —— 它们在 cl100k_base 里是 **2 token**，压了反而亏。
- **现成可用的 38 个符号**：
  `§ ¶ ● ★ ☆ ♥ ♀ ♪ ← ↑ → ↓ ─ ━ │ █ ░ † ¬ ± α β γ δ ε η θ κ λ μ π ρ σ τ υ φ χ ω`
- 现成示例：`test_dict.txt` / `test_dict.json`。

---

## 什么时候别用

- **短文本、一次性问答**：图例成本摊不薄，`--stats` 的回本轮数会 >1。
- **创意写作**：符号化会伤表达。
- **低频内容**：压不掉"低频但占地方"的段落，那部分该交给摘要 / RAG。
- **重复度低的长文**：先看回本轮数，别凭感觉。

中文替换是**按字面**做的、不认词边界，把词典套到领域外文本上时，先跑一次 `--expand` 校验。

---

## 选词逻辑（为什么词典不脏）

按 `出现次数 × (原 token 数 − 1)` 收益降序，逐条过五道筛：

1. **子串**：长短语已覆盖的子串不要；
2. **4 字窗口**：中文与已选条目共享 ≥4 字窗口的跳过；
3. **极大性**：若某短语每次出现都带相同相邻字、且更长串出现次数相同，说明它只是更长重复单元的一截（`言模型是一种模型` ← `语言模型是一种模型`），跳过；
4. **原文位置跨度**：在原文里出现位置与已选条目重叠的一律跳过（否则会选出 `练数据决定` 这种跨边界碎片）；
5. **收益覆盖成本**：单条收益必须盖过它自己那行图例的成本。

英文额外拒绝跨句边界的词组（第二个词起首字母大写的，如 `data The context`）。符号池在启动时逐个校验 `n_tokens(sym) == 1`。

没有这些守卫的话，会选出 `言模型是一种模型`、`练数据决定` 这种半截词——它们看着省，其实不可复用。

---

## 实测（tiktoken cl100k_base）

| 场景 | 命令 | 结果 |
|---|---|---|
| 英文样本 696 token | `context_compactor_demo.py` | → **524**，省 **24.7%**，回本 0.13 轮 |
| 中文样本 186 token | `compact.py --stats` | → **143**，省 **23.1%**，回本 0.40 轮 |
| 多轮对话 6 轮（默认重建） | `--rounds 6` | 612 → 407 tok，省 **33.5%** |
| 多轮 + 固定词典（4 轮） | `predefined_dict=test_dict.txt` | 408 → 332 tok，省 **18.6%** |
| 多轮 + 补词（4 轮） | `allow_grow=True` | 4 → 8 条，省 **25.0%** |

压缩率上限取决于"文本中有多少比例由重复短语构成"。术语 / 专有名词 / 代码标识符高频重复时，压到 30%~50% 很常见。

---

## 常见坑

| 症状 | 原因 | 处理 |
|---|---|---|
| 报 `TokenizerMissing` | 没装 tiktoken | `pip install -r requirements.txt` |
| 省 0%、字典为空 | 用了 `--allow-proxy` 或没装 tiktoken | 同上 |
| 图例里有 `◆ ▲` 这类 | 手写词典用了 2-token 符号 | 换成本文列的符号，看 `sess.warnings` |
| 压完模型答得更差 | 压了当前问题 | `keep_raw` 调大；只压历史 |
| 省了 token 但账单没降 | 默认模式图例逐轮变化，prompt caching 失效 | 改用固定词典模式 |
| MB 级语料跑得慢 | 每条候选要做正则扫描 | 分块压缩（实测 18K 中文字符 0.07s） |

---

## 目录结构

```
scripts/compact.py              Skill A 规范实现（手动压缩 + --expand 还原）
scripts/auto_compactor.py       Skill B 规范实现（请求拦截层）
auto_compactor.py               兼容垫片，转调 scripts/auto_compactor.py
context-compactor/              Skill A 技能说明（SKILL.md，打包进 zip）
context-auto-compactor/         Skill B 技能说明（SKILL.md，打包进 zip）
context-compactor.zip           可分发技能包（SKILL.md + scripts/ + LICENSE）
context-auto-compactor.zip      同上
tests/selftest.py               26 项回归自测
build_packages.py               确定性重建两个 zip（固定时间戳）
context_compactor_demo.py       教学 demo（直接调用规范实现）
test_dict.txt / .json           词典格式示例
思路.md                         设计思路、算法、风险与修复记录
```

## License

GPL-3.0-or-later，见 [LICENSE.txt](LICENSE.txt)。
