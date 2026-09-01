# 一步步跑 ChemBench

从零到拿到分数。每一步都写了**预期输出长什么样**，对不上就知道卡在哪。

---

## 需要什么

| | 大小 | 说明 |
|---|---|---|
| GPU | 显存 ≥60 GB | 27B 模型 bf16 推理 |
| 磁盘 | ≥80 GB | 底座 54 GB + 题库 + 结果 |
| 时间 | 约 30 分钟 | 纯模型 6 分钟 / 接工具 25 分钟 |

显存不够可以用 `device_map="auto"` 多卡切分，或量化加载（会影响分数，需注明）。

---

## 第 1 步 · 部署（一条命令）

```bash
git clone https://github.com/lynnquams/-.git ~/clearchem
cd ~/clearchem
WITH_BENCH=1 WITH_QWEN=1 bash scripts/deploy.sh
```

这一条会自动做完：

```
探测环境 → 装依赖 → 合并权重分卷 → 下 Qwen3.8-27B 底座（54 GB）
        → 下 ChemBench 题库（2785 题）→ 装官方计分库 chembench → 自检
```

底座下载最久（30 分钟到 2 小时，看网速），**断点续传**，中断了重跑同一条命令。

### 确认这一步成功

```bash
ls ~/clearchem/bases/qwen/config.json          # 底座
ls ~/clearchem/data/chembench_hf/              # 题库，应有多个类别目录
python3 -c "import chembench; print('计分库 OK')"
ls ~/clearchem/models/clearchem-qwen/adapter_model.safetensors   # 应为 304 MB
```

四条都有输出才能往下走。

---

## 第 2 步 · 跑裸底座基线

**先跑基线**，这是判断适配器有没有起作用的唯一参照。

```bash
cd ~/clearchem && export CLEARCHEM_ROOT=$PWD
OUT=$PWD/results/cb_base.json python3 scripts/eval_chembench.py
```

预期输出：

```
题库 2785 题  {'multiple_choice_grade': 2542, 'mae': 243}
   504/2785  running=0.5635  1.0min  eta 5min
  ...
官方口径 fraction_correct = 0.5964  (1661/2785)
抽不出答案 0 条（已按官方计为错）
```

**关键检查**：`抽不出答案 0 条`。如果这个数不是 0，说明答案抽取有问题，
后面的分数全部无效 —— 先查提示词模板是不是套上了 `apply_chat_template`。

约 6 分钟。

---

## 第 3 步 · 跑纯模型（加载适配器，不接工具）

```bash
ADAPTER=$PWD/models/clearchem-qwen \
OUT=$PWD/results/cb_pure.json \
python3 scripts/eval_chembench.py
```

预期：

```
已加载 adapter: /root/clearchem/models/clearchem-qwen
官方口径 fraction_correct = 0.6316  (1759/2785)
```

**这就是纯模型成绩。** 分项应为：选择题 0.6475、数值题 0.4650。

---

## 第 4 步 · 跑 agent 系统（接 Python 工具）

```bash
ADAPTER=$PWD/models/clearchem-qwen TOOL=1 BATCH=16 \
OUT=$PWD/results/cb_final.json \
python3 scripts/eval_chembench.py
```

预期：

```
工具调用：243 道数值题让模型写 Python 现算
  代码跑不出 47 道（保留直答）  修好 50  弄坏 6  净 +44
官方口径 fraction_correct = 0.6445  (1795/2785)
```

约 25 分钟（数值题要多跑一轮写代码）。

**"修好 N 弄坏 M" 这两个数要看**：净值为负说明工具在帮倒忙，
应该退回纯模型口径。

---

## 第 5 步 · 统计检验（不能只看分数差）

```bash
python3 scripts/mcnemar.py base:final base:pure pure:final
```

预期：

```
base→final    0.5964 → 0.6445 (+0.0481)  翻对318 翻错184  p=0.0000  **显著**
base→pure     0.5964 → 0.6316 (+0.0352)  ...              p<0.05    显著
pure→final    0.6316 → 0.6445 (+0.0129)  ...              工具的贡献

典型不一致题数 267 → 本基准最小可分辨差 ≈ 0.0115（1.15 个百分点）
```

**为什么必须做这一步**：这套基准的最小可分辨差是 **1.15 个百分点**。
低于这个幅度的变化测不出来，报准确率差值会把噪声当成果。
我们自己曾把 +0.47pp（p=0.46）当作改进报过一次。

---

## 第 6 步 · 看分项（可选但推荐）

```bash
python3 - <<'EOF'
import json
from chembench.metrics import all_correct
d = json.load(open("results/cb_final.json"))
cat = {}
for r in d["results"]:
    cat.setdefault(r["category"], []).append(r["correct"])
for k, v in sorted(cat.items(), key=lambda x: -len(x[1])):
    print("%-24s %4d题  %.4f" % (k, len(v), sum(v)/len(v)))
EOF
```

预期（最终版）：

```
chemical_preference      1000题  0.6610
toxicity_and_safety       675题  0.4578
organic_chemistry         429题  0.7786
physical_chemistry        165题  0.6909
analytical_chemistry      152题  0.5921
general_chemistry         149题  0.8926
...
```

分项能看出改进落在哪里。**总分被 60% 的无关题稀释**，
分项涨 8pp 在总分上可能只体现 3pp。

---

## 常见问题

### 分数远低于预期（比如 0.1 量级）

八成是**没套对话模板**。Qwen 是 instruct 模型，裸提示词等于瞎猜。
检查 `eval_chembench.py` 里有没有：

```python
tok.apply_chat_template([{"role": "user", "content": q}],
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
```

我们踩过这个坑：漏了模板，成绩从 0.5964 掉到 0.1174。

### 抽不出答案的题很多

`max_new_tokens` 太小把模型的回答截断了。本协议用 24 token 并在提示词里
明确要求"只回字母"。如果改成让模型先推理，需要 1024+ token
且仍有 12% 抽不出 —— 实测那样做分数不升反降。

### 数值题工具那一轮很慢

正常。243 道题每道生成 512 token 的代码，再逐个执行（20 秒超时）。
单卡约 8 分钟。想跳过就不加 `TOOL=1`。

### 想开思考模式

```bash
THINK=1 MAXNEW=1024 python3 scripts/eval_chembench.py
```

**实测会掉 7.0 个百分点（p=3e-24）**。模型的"思考"是自言自语不是结构化推理，
对知识题反而把第一直觉里正确的答案推翻了。详见 docs/FAILURES.md。

---

## 结果对照表

跑完应该拿到这些数（单卡 A800，官方 `chembench.metrics.all_correct` 计分）：

| 配置 | 总分 | 选择题 | 数值题 |
|---|---|---|---|
| 裸底座 | 0.5964 | 0.6117 | 0.4362 |
| + 适配器（纯模型） | 0.6316 | 0.6475 | 0.4650 |
| + 适配器 + Python 工具 | 0.6445 | 0.6475 | 0.6049 |

对不上的话，先查 `抽不出答案` 是不是 0，再查有没有套对话模板。
