# 复现指南

**克隆仓库能跑平台功能，但不能直接跑分。** 复现 ChemBench 0.6445 还需要三样东西，
本文说明怎么补齐。

---

## 克隆后能跑什么、不能跑什么

| | 需要什么 | 克隆后是否可用 |
|---|---|---|
| 性质预测 | 打分器权重（仓库里有） | **可以** |
| 配方推荐 | 电导率打分器 + CALiSol 数据 | **可以**（deploy.sh 自动下数据） |
| 分子生成 | 生成器 LoRA + ether0 底座 48 GB | 需下底座 |
| 条件遵循评测 | 同上 | 需下底座 |
| **ChemBench 跑分** | **Qwen 底座 54 GB + 题库 + 评测脚本** | **需自行准备前两样** |

---

## 一、复现 ChemBench 0.6445

### 1. 一条命令准备全部依赖

```bash
cd ~/clearchem && WITH_BENCH=1 WITH_QWEN=1 bash scripts/deploy.sh
```

自动完成：Qwen3.8-27B 底座（54 GB）· ChemBench 题库（2785 题）· 官方计分库 `chembench`。
断点续传，中断了重跑同一条命令。

脚本还会校验底座与适配器的维度是否匹配——下错版本会当场报错，不会等到跑分时才发现。

**必须用官方 `chembench.metrics.all_correct` 计分**，不要自己写判分逻辑。
我们自己复现的判分曾因浮点尾数问题使成绩虚高 0.25pp
（工具算出 `-0.6000000000000005`，官方要求与 `-0.6` 严格相等）。

### 4. 跑评测

```bash
cd ~/clearchem
export CLEARCHEM_ROOT=$PWD

# 裸底座基线（预期 0.5964）
python3 scripts/eval_chembench.py

# 加载适配器 + Python 工具（预期 0.6445）
ADAPTER=$PWD/models/clearchem-qwen TOOL=1 BATCH=16 \
  OUT=$PWD/results/cb_final.json \
  python3 scripts/eval_chembench.py
```

单卡 A800 约 25 分钟（含数值题写代码那一轮）。

### 5. 统计检验

```bash
python3 scripts/mcnemar.py base:final
```

输出应为：

```
base→final    +0.0481   翻对318  翻错184   p=0.0000  **显著**
```

**只报准确率差值是不够的。** 这套基准的最小可分辨差是 1.15 个百分点
（MDE ≈ 1.96·√n_discordant / n_total），低于此幅度的变化测不出来。

---

## 二、复现条件生成 MAE 0.109

需要 ether0 底座（`deploy.sh` 会自动下）。

```bash
cd ~/clearchem && export CLEARCHEM_ROOT=$PWD
python3 scripts/eval_condition.py
```

预期输出：

```
  目标 5.0   K= 6 T=1.2  实测均值 4.99  (偏差-0.01)  MAE 0.073
  目标 6.5   K= 6 T=1.2  实测均值 6.53  (偏差+0.03)  MAE 0.069
  目标 8.0   K= 8 T=1.2  实测均值 8.02  (偏差+0.02)  MAE 0.061
  目标 9.5   K=10 T=1.3  实测均值 9.50  (偏差+0.00)  MAE 0.122
  目标 11.0  K=14 T=1.4  实测均值 10.98 (偏差-0.02)  MAE 0.263
  目标 12.5  K=20 T=1.4  实测均值 12.47 (偏差-0.03)  MAE 0.065

  分区拒绝采样  MAE 0.109  Spearman 0.986
```

单卡约 30 分钟。

---

## 三、复现打分器

打分器可以从头重训（数据需自备带 DFT 标签的分子）。**两条纪律必须遵守**：

### 分子性质：按 Murcko 骨架切分

```python
from rdkit.Chem.Scaffolds import MurckoScaffold
key = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
# 同骨架分子整块进同一侧，不能随机切
```

同骨架分子性质高度相关，随机切会泄漏。

### 配方电导率：按文献切分 + 必做交叉验证

```python
# 按 doi 分组，且必须报 K 折交叉验证的均值±方差
```

实测同一模型：随机切 R² 0.8859、按文献单次切 0.7336、5 折交叉验证 −0.03/−inf/0.14/0.29/−0.20。
**数据源少于 30 个时，单次切分的数字没有意义。**

---

## 四、评测纪律（踩过坑总结）

1. **任何自研评测工具，用之前先量它自己的误差**，并与待测量级对比。
   我们的 gap 打分器初版 MAE 1.698 eV，**比它要测的对象误差还大**，
   用它得出的四轮训练结论全部作废。
2. **新评测协议必须先复现旧协议**，写成断言。
   曾因漏 `apply_chat_template` 使成绩从 0.5964 掉到 0.1174 而险些误判。
3. **报提升前做配对 McNemar**，不看准确率差值。
4. **先算最小可分辨差**，低于此幅度的改动不能当改进。
5. **分组数 < 30 时报 K 折交叉验证**，不报单次切分。

详见 [FAILURES.md](FAILURES.md)。

---

## 五、已知无法直接复现的部分

| 项目 | 原因 |
|---|---|
| 训练数据 | MolSkill 需自行 clone（MIT，`github.com/microsoft/molskill`），去重脚本见 TECHNICAL.md |
| 蒸馏数据 | 由自训 RankNet 标注 PubChem 生成，脚本未开源（可按 TECHNICAL.md 1.2 节重建） |
| MLIP 部分 | MACE-OMol 在电解液输运性质上不可用（扩散慢 2500 倍），见 BENCHMARKS.md 第 4 节 |
