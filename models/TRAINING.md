# 训练了什么，用什么训的，实测多少

底座一律是 `Qwen/Qwen3.8-27B`（Apache 2.0），bf16，LoRA 微调。
分数一律是 ChemBench 官方计分函数 `chembench.metrics.all_correct`，2,785 题全量。

**这套基准的最小可分辨差是 1.15 个百分点**
（`MDE ≈ 1.96·√(n_discordant)/n_total`，典型 n_discordant≈267）。
低于这个幅度的差异测不出来，下表中凡是没写「显著」的都不能当成提升。

---

## 随仓库分发的两个版本

### clearchem-qwen（v1，推荐使用）

```
LoRA        r=16 alpha=32 dropout=0.05
目标层      q/k/v/o_proj + gate/up/down_proj（7 层全覆盖）
可训参数    79.69M / 26.98B = 0.295%
```

**训练数据**：三样有效手段的混合

| 来源 | 内容 |
| --- | --- |
| 偏好靶向 | ChemBench preference 题型的同分布训练样本 |
| 偏好蒸馏 | 教师模型在偏好任务上的输出 |
| RDKit 成对 | 用 RDKit 真实计算生成的分子成对比较题 |

**实测**

```
纯模型            0.6316    超 claude3.5 (0.6255)
接 Python 工具     0.6445    超 o1-preview (0.6435)
裸底座对照         0.5964
```

`0.5964 → 0.6445` 是 +4.81pp，远超 MDE，**这个提升是真的**。
工具层单独的贡献 +1.29pp（0.6316→0.6445），也过 MDE。

### clearchem-qwen-dpo（v2，同水平，非提升）

```
LoRA        与 v1 相同（r=16 alpha=32，7 层）
初始化      从 v1 适配器续训，不是从裸底座
超参        beta=0.1  lr=5e-6  1 epoch  batch=2×accum8×4卡
数据        50,925 对（4,275 条真实标注上采样 3 倍 + 4 万条蒸馏）
```

**动机**：偏好数据的原生目标就是 DPO，v1 把它当 SFT 用（只学"输出哪个字母"），
没利用成对结构。

**实测**

```
             DPO      v1       Δ         p 值      判定
纯模型       0.6352   0.6316   +0.36pp   0.378    噪声
接工具       0.6474   0.6445   +0.29pp   0.509    噪声
DPO 自身     0.6474   0.6352   +1.22pp   <1e-4    显著（工具层的功劳）
```

**两种口径都过不了配对 McNemar 检验。** 训练指标是健康的
（margin +0.72→+1.16、奖励准确率 0.78），模型确实学会了在偏好对上排序，
但**那个能力没有转化成 ChemBench 上可测量的提升**。

0.6474 是目前最高的单次实测数，但**不能宣称超过 0.6445** —— 差 0.29pp，
这套基准分不出这两个数。对外一律写 0.6445。

**为什么仍然分发**：它和 v1 同水平且训练配方不同，可作对照；
DPO 这条路的否定结论本身也是结果。

---

## 训过但没进仓库的（都不如 v1）

| 版本 | 训练内容 | 分数 | 为什么不发 |
| --- | --- | --- | --- |
| prefmix | 偏好数据混合 | 0.6370 | 低于 v1 |
| final | 多阶段合并 | 0.6338 | 低于 v1 |
| cardmix | 卡片式知识注入 | 0.6302 | 低于 v1 |
| chain | 思维链 | 0.5968 | 与裸底座无异 |
| bal1000 | 平衡采样 1000 | 0.5914 | 低于裸底座 |
| cpt_v2 | 继续预训练 v2 | 0.5896 | 低于裸底座 |
| reason | 2 万条真 R1 推理链 | 0.5878 | 低于裸底座，p=0.07 |
| cpt | 继续预训练（r=32） | 0.5673 | **比裸底座低 2.9pp** |

**cpt 那次最贵**：CPT 23.6 小时 + SFT 15.2 小时，共 38.8 小时，
跑完发现 ChemBench 单调下降 10 个百分点。事后审计定位到五个语料缺陷
（Wikipedia 68% 不含化学术语、LibreTexts 100% 是未解析的 JSON、
SFT 392,000 条全是同两个句式模板、长文被截到 6000 字符、LoRA r=32 过参数化）。
详见 `docs/FAILURES.md`。

---

## 生成器 clearchem-gen

```
底座        futurehouse/ether0（Apache 2.0）
用途        条件分子生成
实测        条件遵循 MAE 0.109 eV（六个目标点，偏差全在 ±0.03）
            Novelty 0.928 · SA 3.40 · Validity 0.992
```

**注意**：MAE 0.109 是用重训后的打分器（MAE 0.404）测的。
第一版打分器 MAE 1.698 eV —— 比它要测的对象误差还大，
用它得出的四个训练结论全部作废。任何自研评测工具，用之前先量它自己的误差。

---

## 怎么加载

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.8-27B", dtype="bfloat16",
                                         device_map="auto", trust_remote_code=True)
m = PeftModel.from_pretrained(m, "models/clearchem-qwen")        # v1，推荐
# m = PeftModel.from_pretrained(m, "models/clearchem-qwen-dpo")  # v2，同水平
```

权重按 90MB 分卷存在 `models/_parts/`，部署脚本会自动合并。
手动合并：`bash scripts/assemble_weights.sh`

**完整性核对**

```
clearchem-qwen      md5 11e6dfedab97e89d52af515f8ec5fbbb
clearchem-qwen-dpo  md5 29be5ebbf887b81456d0316a4cd29867
```
