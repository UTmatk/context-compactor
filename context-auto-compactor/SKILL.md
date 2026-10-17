---
name: context-auto-compactor
description: 把"高频词/短语→单字符"的确定性词典压缩做成透明请求拦截层，让长对话/长文档在每轮调用模型前自动压缩上下文，对调用方无感。当用户说"自动压缩上下文""做成自动拦截层""每轮自动省 token""把上下文压缩做成中间件"或希望长对话不被上下文窗口拖垮、又不想每轮手动触发时使用。提供可运行的 scripts/auto_compactor.py（含离线 dry-run 演示，支持中英文混合），也支持接入真实 OpenAI 兼容客户端与 Skill A 预置词典。
agent_created: true
---

# 上下文自动压缩拦截层

## Overview

「context-compactor（手动 Skill）」的自动版：把同一条确定性词典压缩做成**透明中间件**。每轮 `chat()` 自动完成"用全量历史重建字典 → 压缩除最近 N 条用户消息外的所有历史 → 映射表拼进 system 消息 → 发送 → 存原文"，调用方仍照常调用 `chat()`。历史越滚越大，压缩省得越多。

- 适用：多轮长对话、长文档问答等上下文持续膨胀的场景。
- 不适用：超短对话（图例成本摊不薄）、要求逐字保真的场景。

## 依赖

`tiktoken` **是硬依赖**。缺了它 `build_mapping()` 抛 `TokenizerMissing`、CLI 打印可操作错误并退出，不会静默产出一份空字典。

```bash
pip install -r scripts/requirements.txt
```

## Workflow

### Step 1: 确定接入方式
- **离线演示 / 快速验证**：直接跑脚本，无需 API key。
- **接入真实客户端**：把 `transport` 设为 `openai_client.chat.completions.create`，之后用 `sess.chat(...)` 代替原来的请求调用。

### Step 2: 运行（离线演示）
```bash
python scripts/auto_compactor.py --rounds 6
python scripts/auto_compactor.py --rounds 6 --dict my_dict.txt        # 固定词典
python scripts/auto_compactor.py --rounds 6 --dict my_dict.txt --grow # 固定词典 + 补词
```
逐轮打印「字典条数 / 原文历史 token / 实际发送 token / 节省比例」，并显示 system 消息里的图例。前几轮图例未摊薄为负属正常，之后转正并随历史增长。

### Step 3: 接入真实客户端
```python
from openai import OpenAI
from auto_compactor import AutoCompactSession

client = OpenAI()
sess = AutoCompactSession(transport=client.chat.completions.create)
sess.system = "你是一个助手"
reply = sess.chat("用户的第一句话", model="gpt-4o")   # 内部自动压缩后发送
```
`keep_raw` 控制最近几条用户消息保持原文；`cjk_max` 控制中文最大词长（默认 12）；`top` 控制字典条数上限。

### Step 4: 查看收益
`sess.stats()` → `{"dict_size", "history_tokens", "sent_tokens", "saved", "saved_pct"}`。

## 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `transport` | `DryRunTransport` | 发送函数；真实场景传 `client.chat.completions.create` |
| `top` | 15 | 字典条数上限（受单 token 符号数量封顶） |
| `max_ngram` | 3 | 英文词组最大词数 |
| `cjk_min` / `cjk_max` | 2 / 12 | 中文词串长度区间 |
| `keep_raw` | 1 | 最近几条用户消息保持原文 |
| `verbose` | False | 逐轮打印压缩明细 |
| `predefined_dict` | None | 预置词典（dict 或路径字符串），非空即进入固定字典模式 |
| `allow_grow` | False | 固定词典之外是否每轮补发现新词 |
| `allow_proxy` | False | 无 tiktoken 时也跑（计数不可信） |

## 预置词典（A 建库 → B 套用，推荐给稳定术语库）

```python
sess = AutoCompactSession(predefined_dict="dict.txt")            # 构造时传入
sess.load_dictionary("dict.txt")                                # 或运行时加载
sess = AutoCompactSession(predefined_dict="dict.txt", allow_grow=True)  # 再补新词
```

接受三种文件：

1. **图例文本**：`{符号} = {短语}` 一行一条，`Dictionary ...` 头与空行忽略。
2. **Skill A 直接产出的 packed 文件**：**只解析图例段**，第一个空行之后的压缩正文整体跳过。
   （旧版会把正文里任何含 `=` 的行当成词典条目，例如 `train = fit` 会变成一条映射，
   进而把正文里的 `fit` 替换成 `train` —— 已修。）
3. **JSON**：`{短语: 符号}` 或 `{符号: 短语}`，按首个 key 自动判断方向。

加载后 `sess.warnings` 会给出手写词典的体检结果（符号是否 1 token、是否重复分配）。

> 离线验证（4 轮中文，仅 4 条预置词条）：固定字典模式 18.6%；补词模式追加到 8 条、25.0%。预置词典越全，覆盖越高。

## 关键约束

1. **替换符必须单 token**：候选池启动时用 tokenizer 校验，只保留 `n_tokens == 1` 的符号（cl100k_base 下 38 个）。
2. **词典成本要摊销**：自动版靠"历史增长"摊薄图例成本；短对话勿用。
3. **最近用户消息保持原文**：`keep_raw` 默认 1，避免压缩当前问题影响推理。
4. **失败回滚**：传输函数抛异常时本轮用户消息会从 history 中回滚，不留没有回复的半截轮次。
5. **图例每轮都变（默认重建模式）**：每次请求都随附当前图例，模型侧自洽；但这意味着 system 消息逐轮变化，**prompt caching 会失效**。长稳定会话建议改用固定词典模式（`predefined_dict`），图例跨轮恒定。
6. **CJK 选词**：三道去重（子串 / 4 字窗口 / 原文位置跨度）+ 极大性判据，避免整句重复时选出「言模型是一种模型」这类碎片。

## 与其他方案的关系
与「滑窗 / RAG / 摘要」互补：高频术语走符号映射（确定性、可逆、零额外推理），低频大块内容仍靠 RAG/摘要兜底。

## 实现说明
`scripts/auto_compactor.py` 内：`build_mapping` 统计中英文 n-gram → 收益降序 → 五道筛选 → 分配 1-token 符号；`AutoCompactSession._outgoing` 建字典并压缩历史；`DryRunTransport` 提供离线演示传输层；`expand()` 提供逆运算。算法与 `context-compactor` 的 `compact.py` 保持一致，`tests/selftest.py` 会校验两边输出相同。

## License

Copyright (C) 2026 UTmatk — **GPL-3.0-or-later**，全文见同目录 `LICENSE`。
本文档与 `scripts/auto_compactor.py` 同许可。
