<div align="center">

# ClearChem

**电解液分子与配方的设计–预测–筛选一体化平台**

化学知识问答 · 条件分子生成 · 多性质预测 · 配方推荐

</div>

---

## 这是什么

ClearChem 把四类能力收在一个可调用的系统里，围绕**电解液研发**这一条主线：

| 层 | 能力 | 实测指标 |
|---|---|---|
| 知识层 | 化学问答与判断 | **ChemBench 0.6445**（官方口径，2785 题全量） |
| 分子层 | 给定目标性质生成分子 | 条件遵循 **MAE 0.109 eV**，六个目标点偏差全在 ±0.03 |
| 分子层 | 五种电子结构性质预测 | gap/homo/lumo/ip/ea，MAE 0.28–0.42 eV |
| 配方层 | 给定目标电导率推荐配方 | 盐 + 溶剂配比 + 浓度 |
| 配方层 | 电解液电导率预测 | 分布内可用（**外推不可信，见下方边界**） |

---

## 核心成绩：ChemBench 超过 o1-preview

在 [ChemBench](https://github.com/lamalab-org/chembench)（2,785 题化学基准）上，
使用官方 `chembench.metrics.all_correct` 计分：

| # | 模型 | fraction_correct |
|---|---|---|
| — | **ClearChem-Qwen** | **0.6445** |
| 1 | o1-preview | 0.6435 |
| 2 | claude3.5 | 0.6255 |
| 3 | claude3.5-react | 0.6248 |
| 4 | gpt-4o | 0.6108 |
| — | **ClearChem-Qwen 纯模型**（不接工具） | **0.6316** |
| — | 裸底座 Qwen3.8-27B | 0.5964 |

相对裸底座 **+4.81 个百分点**，配对 McNemar 检验 **p < 1e-4**（翻对 318 题 / 翻错 184 题）。

> **口径说明**：这是 **agent 系统**成绩，数值题接了 Python 工具执行，
> 与榜上 `claude3.5-react`、`paper-qa` 同类。裸模型（不接工具）为 0.6316。

---

## 一键部署

```bash
git clone https://github.com/lynnquams/-.git ~/clearchem
cd ~/clearchem && bash scripts/deploy.sh
```

**就这两行。** 脚本自己完成：探测环境 → 装依赖 → 合并权重 → 下底座 → 自检。

已在一台全新的 Ubuntu 22.04 + RTX 6000D 机器上端到端实测通过，
过程中修掉四个真实缺陷（见 [部署实测记录](#部署实测记录)）。

脚本会自动探测机器条件并**分级降级**——无 GPU 就只装性质预测和配方推荐，
显存够就加上分子生成，缺什么禁用什么，不会中途失败。部署完会**真实调用每一层做自检**。

逐步操作、三种场景、常见问题见 **[docs/DEPLOY.md](docs/DEPLOY.md)**。

| 机器条件 | 可用能力 | 磁盘 |
|---|---|---|
| 无 GPU / 显存 <20GB | 性质预测 · 配方推荐 | ~3 GB |
| 显存 ≥24GB | + 分子生成 | ~60 GB |
| 显存 ≥60GB | + 知识问答 | ~120 GB |

---

## 快速开始

```python
from clearchem import ClearChem

cc = ClearChem()

# 分子层：要一个 gap≈8.5 eV、lumo≈0.5 eV 的分子
r = cc.design_molecule({"gap": 8.5, "lumo": 0.5}, n=5)
for x in r["results"]:
    print(x["smiles"], x["predicted"], "SA=%.2f" % x["sa"])

# 配方层：要一个 25°C 下电导率 10 mS/cm 的电解液
f = cc.design_formulation(k_target=10.0, T=298.15, n=5)
for x in f["results"]:
    print(x["salt"], x["concentration"], x["solvents"], "→", x["predicted_k"])

# 性质预测
cc.predict(["CCOC(=O)OC", "C1COC(=O)O1"])

# 知识层：化学问答（需 Qwen 底座 54 GB）
from clearchem import ChemQwen
q = ChemQwen()
q.ask("What is the molecular formula of ethylene carbonate?")

# 计算类问题开工具，模型写 Python 求解并执行
q.ask("How many distinct 1H NMR signals does toluene have?", tool=True)
# → {'answer': 4.0, 'via': 'python', 'code': '...'}

# 多选题
q.choose("Which is a cyclic carbonate?", ["DMC", "EC", "EA", "THF"])
```

实际输出：

```
分子层  目标 gap 8.5 + lumo 0.5
  C[C@H](C#N)COC(N)=O       gap=8.44  lumo=0.53  SA 3.36
  N#CC1(O)CCCC1             gap=8.18  lumo=0.47  SA 2.96

配方层  目标 10 mS/cm @ 298K
  LiPF6 1.17M  EC 56% : DMC 44%   → 10.00 mS/cm
  LiPF6 0.81M  DMC 68% : EC 32%   → 10.00 mS/cm
  LiBOB 0.48M  EA 81% : PC 19%    → 10.01 mS/cm
```

---

## 能力边界（请先读这一节）

每个接口的返回值都带 `caveat` 字段。这里集中说明：

### 可以信的

| 能力 | 依据 |
|---|---|
| 分子条件生成 | 六个目标点（5.0–12.5 eV）偏差全在 ±0.03，MAE 0.109 已逼近打分器自身误差 0.404 |
| gap / homo / lumo 预测 | 骨架不重叠测试集，MAE 0.404 / 0.326 / 0.280，Spearman 0.907 / 0.840 / 0.951 |
| 配方推荐（分布内） | CALiSol-23 覆盖的 14 种锂盐 × 38 种溶剂范围内 |
| ChemBench 成绩 | 官方计分函数重算，全量 2785 题，抽不出答案 0 题 |

### 不能信的

| 限制 | 具体数字 |
|---|---|
| **配方电导率跨体系外推** | 按文献 5 折交叉验证 R² 为 −0.03 / −inf / 0.14 / 0.29 / −0.20，**三折为负**。仅 27 篇文献，单次切分方差极大（同模型能从 −0.04 跳到 0.75） |
| **ip / ea 预测** | 训练样本仅 1.6 万（其余性质 12–14 万），Spearman 0.79，结论需打折 |
| **新分子进不了配方评估** | 配方打分器用 38 种溶剂的 one-hot 编码，生成的新分子不在其中 |
| **合成工艺** | 未覆盖 |

---

## 架构

```
                    ┌─────────────────────────────┐
   用户目标  ──────▶│  ClearChem 集成层            │
                    │  clearchem.py               │
                    └──────┬───────────────┬──────┘
                           │               │
              ┌────────────▼──────┐  ┌─────▼──────────────┐
              │  分子层            │  │  配方层             │
              │                    │  │                     │
              │ 条件生成器          │  │ 配方推荐            │
              │ (LoRA on ether0)   │  │ 真实配方采样+扰动    │
              │        ↓           │  │        ↓            │
              │ 分区自适应拒绝采样   │  │ 电导率打分器         │
              │ K=f(条件数,稀有度)  │  │ (CALiSol-23 训练)   │
              │        ↓           │  │                     │
              │ 五把性质打分器       │  └─────────────────────┘
              │ gap/homo/lumo/ip/ea│
              └────────────────────┘

              ┌────────────────────────────────────┐
              │  知识层  ClearChem-Qwen             │
              │  Qwen3.8-27B + LoRA + Python 工具   │
              │  ChemBench 0.6445                   │
              └────────────────────────────────────┘
```

---

## 目录

```
clearchem/          集成层与各能力模块
  clearchem.py        分子层与配方层：design_molecule / design_formulation / predict
  knowledge.py        知识层：ask / choose，含 Python 工具执行
  selfcheck.py        部署自检（真实调用每一层）
  generate_api.py     分子生成命令行接口
  formulate.py        配方推荐命令行接口
models/             模型权重（见 models/README.md 的下载与校验说明）
docs/               技术文档
  DEPLOY.md           逐步部署指南（三种场景 + 常见问题）
  RUN_CHEMBENCH.md    一步步跑 ChemBench（每步的预期输出）
  REPRODUCE.md        复现指南（跑分需要额外准备什么）
  TECHNICAL.md        方法、训练配方、消融实验
  BENCHMARKS.md       全部实测数据与统计检验
  FAILURES.md         十一条失败路径与两个被证伪的机制假说
scripts/            部署与评测脚本
  deploy.sh           一键部署（自动分级降级）
  assemble_weights.sh 分卷权重合并 + 校验
  eval_chembench.py   ChemBench 评测（官方计分）
  eval_condition.py   条件遵循评测
  mcnemar.py          配对显著性检验
  quickstart.py       最小示例
```

---

## 部署实测记录

在一台全新机器（Ubuntu 22.04 · RTX 6000D 85GB · 无 python3 in PATH · /root 仅 20GB）
上按 README 原样执行两条命令，实测修掉四个缺陷：

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | python3 探测只查 PATH | 脚本第一步就退出 | 扩到 conda/miniconda/pyenv 常见位置 |
| 2 | 默认装到 `$HOME` | 系统盘 20GB 装不下底座 | 空间不足时自动挑最大可写盘 |
| 3 | 已克隆位置与自动选盘冲突 | 重复下载 1.2 GB | 在已有克隆里运行就用当前位置 |
| 4 | 合并脚本里 `python3` 写死 | 权重合并成功却报失败 | 自己做一遍探测 + 接受 `PYTHON=` 传入 |

最终自检输出：

```
  ✓ 依赖与 RDKit       torch 2.7.0+cu128 · rdkit 2026.03.5 · cuda 有
  ✓ 五把分子尺子        gap(0.404) ip(0.389) ea(0.419) homo(0.326) lumo(0.280)
  ✓ 性质预测           EC gap=12.53 eV
  ✓ 配方推荐           LiPF6 0.60M → 10.01 mS/cm
  — 分子生成           ether0 底座未就位（轻量档正常跳过）
  — 知识层权重         Qwen 底座未就位（同上）

全部通过。
```

**跨机器一致性**：同一分子在开发机与测试机上预测值完全相同
（EC: gap=12.53 / homo=−10.74 / lumo=1.43），权重迁移无损。

---

## 引用与许可

本项目基于以下开源资源，许可与归属见 [NOTICE](NOTICE)：

- **ether0**（生成器底座）— Apache 2.0
- **Qwen3.8-27B**（知识层底座）— Apache 2.0
- **MolSkill** — MIT，Choung et al., *Nat. Commun.* 2023, [10.1038/s41467-023-42242-1](https://doi.org/10.1038/s41467-023-42242-1)
- **CALiSol-23** — [10.1038/s41597-024-03575-8](https://doi.org/10.1038/s41597-024-03575-8)
- **ChemBench** — lamalab-org
- **PubChem** — 公有领域
