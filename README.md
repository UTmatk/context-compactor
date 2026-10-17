# context-compactor

把长文本 / 对话历史里高频重复的短语替换成**单 token 符号**，从而在发给模型前降低 token 占用。确定性、可逆，不依赖小模型。

```text
原文    : 大型语言模型是一种基于神经网络的语言模型。上下文窗口决定……
词典    : § = 大型语言模型   ¶ = 上下文窗口   ● = 训练数据
压缩后  : §是一种基于●的语言模型。¶决定……
```

## 它能做什么

| 形态 | 脚本 | 用法 |
|---|---|---|
| **A 手动压缩** | `scripts/compact.py` | 一次性压缩长文档 / 在语料上产出一份固定词典 |
| **B 自动拦截层** | `scripts/auto_compactor.py` | 包住你的 `chat()`，每轮自动压缩历史，对调用方透明 |

## 快速开始

需要 Python ≥ 3.10 和 `tiktoken`（**硬依赖**，缺了会直接报错，不会静默产出 0% 的假结果）。

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

接入真实模型：

```python
from openai import OpenAI
from auto_compactor import AutoCompactSession        # 或在 scripts/ 下导入

client = OpenAI()
sess = AutoCompactSession(transport=client.chat.completions.create)
sess.system = "你是一个助手"
print(sess.chat("你好", model="gpt-4o"))
print(sess.stats())          # {'dict_size': 7, 'saved_pct': 33.5, ...}
```

推荐用法是 **A 建库 → B 套用**：先用 A 在领域语料上产出一份固定词典，再让 B 加载，字典跨轮恒定（对 prompt caching 友好）。

## 实测（tiktoken cl100k_base）

| 场景 | 结果 |
|---|---|
| 英文样本 696 token | → 524，省 **24.7%** |
| 中文样本 186 token | → 143，省 **23.1%** |
| 多轮对话 6 轮（默认重建） | 612 → 407 tok，省 **33.5%** |
| 多轮对话 + 固定词典（4 轮） | 408 → 332 tok，省 **18.6%** |

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
build_packages.py               确定性重建两个 zip
context_compactor_demo.py       教学 demo（直接调用规范实现）
test_dict.txt / .json           词典格式示例
使用说明.md                     完整操作手册（参数、排坑、实测）
思路.md                         设计思路、算法、风险与修复记录
```

## 选词为什么不选垃圾

按 `出现次数 × (原 token 数 − 1)` 排序后，逐条过五道筛：子串、4 字窗口、**极大性**（是否只是更长重复单元的一截）、**原文位置跨度**（是否与已选条目在原文中重叠）、收益覆盖图例成本。符号池在启动时逐个校验 `n_tokens(sym) == 1`——`◆ ▲ ⚡` 这类其实是 2 token，会被剔除。

没有这些守卫的话，会选出 `言模型是一种模型`、`练数据决定` 这种半截词。

## 注意

- 中文替换按字面进行，不认词边界；把词典套到领域外文本时先跑一次 `--expand` 校验。
- 默认模式每轮重算字典 → system 消息逐轮变化，prompt caching 会失效；长稳定会话请用固定词典模式。
- 不适用：短文本、低频重复、创意写作。

## License

GPL-3.0-or-later，见 [LICENSE.txt](LICENSE.txt)。
