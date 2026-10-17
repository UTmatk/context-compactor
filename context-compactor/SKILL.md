---
name: context-compactor
description: 压缩长文本 / 对话历史 / 提示词的 token 占用，用单字符符号替换高频词与短语（确定性词典编码），在发送给模型前显著缩减上下文。当用户说"压缩上下文""省 token""把这段对话/文档压缩一下""context 太大了""token 超限"或希望缩减长文 token 数时使用。提供可运行的压缩脚本，支持中英文混合，输出压缩文本与映射表，并报告压缩率与回本轮数；支持 --expand 精确还原。
agent_created: true
---

# 上下文压缩（确定性词典编码）

## Overview

把长文本里高频出现的多 token 短语（英文词组、中文词串）映射成**单 token** 符号，前置一份只发一次的映射表（图例），从而缩减发送给模型的 token 数。与 LLMLingua 这类"软"压缩不同，本方法是**确定性、可逆**的：固定映射，`--expand` 可逐字节还原。

- 适用：多轮长对话历史、长文档、术语 / 专有名词 / 代码标识符高频重复的文本。
- 不适用：短文本、低频重复、创意生成（压缩会伤表达）。

## 依赖

`tiktoken`（cl100k_base）**是硬依赖**。缺了它脚本会直接报错退出，而不是静默降级——近似计数（~4 字符/token）会把每个中文短语都算成 <2 token，结果是"字典为空 + 省 0%"，看起来像工作正常。

```bash
pip install -r scripts/requirements.txt   # 或 pip install tiktoken
```

## Workflow

### Step 1: 确定输入
文件路径（`--input file`）或直接粘贴的文本（从 stdin 读）。

### Step 2: 运行压缩脚本

```bash
# 处理文件并查看统计
python scripts/compact.py --input long.txt --output packed.txt --stats

# 管道处理
cat long.txt | python scripts/compact.py --stats

# 还原（读 packed.txt 自带的图例，或用 --dictionary 指定）
python scripts/compact.py --expand --input packed.txt --output restored.txt
```

关键参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--top` | 15 | 最多映射条数（受"单 token 符号"数量封顶，实测 38） |
| `--max-ngram` | 3 | 英文词组最大词数 |
| `--cjk-min` / `--cjk-max` | 2 / 6 | 中文词串长度区间 |
| `--no-legend` | 关 | 省略映射表表头（自行管理映射表时用） |
| `--stats` | 关 | 打印原始 / 压缩 / 图例 token 数与节省比例、回本轮数 |
| `--expand` | 关 | 反向模式：还原压缩文本 |
| `--dictionary` | — | `--expand` 时输入里没有图例，用它指定词典文件 |
| `--allow-proxy` | 关 | 无 tiktoken 时也跑（数字不可信，中文不会选出词条） |

### Step 3: 查看报告
输出：原始 token、压缩后正文、图例成本、总节省比例、回本轮数（盈亏平衡点）、可用符号数。

### Step 4: 交付压缩结果
把"映射表 + 压缩正文"作为新上下文发给模型。若多轮复用，图例放 system prompt / 长期记忆，后续轮次只发符号版历史。

### 进阶：把图例导出给「自动拦截层」复用

本 Skill 产出的 `{符号} = {短语}` 图例，可**直接**作为 `context-auto-compactor` 的预置词典（A 建库、B 套用）：

```bash
python scripts/compact.py --input corpus.txt --output dict.txt --stats
```

```python
from auto_compactor import AutoCompactSession
sess = AutoCompactSession(predefined_dict="dict.txt")   # 固定词典，每轮不重建
```

**不要**把 A、B 串在同一段文本上做双重压缩。

## 关键约束

1. **替换符必须单 token**：启动时用 tokenizer 逐个校验候选池，只保留真正 1 token 的符号（`◆`、`▲`、`⚡` 这类在 cl100k_base 里是 2 token，已剔除）。请勿手改符号池；若必须自建词典，先过一遍 `n_tokens(sym) == 1`。
2. **词典成本要摊销**：图例本身占 token，只在术语高频重复时划算；`--stats` 的"回本轮数"就是盈亏平衡点。
3. **选词的三道去重**（缺一条就会产生垃圾词条）：
   - 子串：长短语已覆盖的子串不再入选；
   - 4 字窗口：中文共享 ≥4 字窗口的条目视为重复；
   - 位置跨度：在原文里出现位置重叠的候选一律跳过（否则会选出「练数据决定」这种跨边界碎片）；
   - 此外还有**极大性**判据：若某个短语的每次出现都带着相同的相邻字、且更长的串出现次数相同，说明它只是更长重复单元的一截，跳过。
4. **只压历史、不压当前问题**：让模型在完整文本上推理新问题。
5. **词典一致性**：同一段会话映射表保持一致，中途不要改。
6. **可还原**：`--expand` 已实现严格逆运算；关键场景建议压缩后立刻跑一次还原做校验。

## 实现说明

`build_mapping`：统计中英文 n-gram（中文多取 1 位用于极大性判据）→ 按 `次数 × (原 token 数 − 1)` 排序 → 逐条过"子串 / 4 字窗口 / 极大性 / 位置跨度 / 收益覆盖图例成本"五道筛 → 分配经校验的 1-token 符号 → `compress` 最长优先替换。符号池按"越不像自然文本越靠前"排序：`§¶●★☆♥♀♪`（历史默认，优先复用）→ `←↑→↓` → `─━│█░` → `†¬±` → 希腊小写字母。

## License

Copyright (C) 2026 UTmatk — **GPL-3.0-or-later**，全文见同目录 `LICENSE`。
本文档与 `scripts/compact.py` 同许可。
