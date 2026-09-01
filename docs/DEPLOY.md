# 部署指南

从零到能用，逐步操作。三种场景按你的机器条件选一个。

---

## 先确认你的机器属于哪一档

在目标服务器上跑这几行：

```bash
python3 --version                                    # 需要 ≥ 3.9
nvidia-smi --query-gpu=name,memory.total --format=csv # 没有这条命令 = 无 GPU
df -h ~ | tail -1                                    # 看可用磁盘
```

| 你的情况 | 能用的功能 | 需要磁盘 | 走哪一节 |
|---|---|---|---|
| 无 GPU，或显存 < 20 GB | 性质预测 · 配方推荐 | ~3 GB | [场景 A](#场景-a轻量部署) |
| 显存 ≥ 24 GB | 上面全部 + 分子生成 | ~60 GB | [场景 B](#场景-b完整部署) |
| 显存 ≥ 60 GB | 上面全部 + 知识问答 | ~120 GB | [场景 C](#场景-c含知识层) |

**不确定就先走场景 A。** 部署脚本会自动探测，条件不够时自动降级，不会失败。

---

## 场景 A：轻量部署

**适用**：笔记本、纯 CPU 服务器、显存不足的机器。能用性质预测和配方推荐。

### 第 1 步 · 装系统依赖

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y git python3 python3-pip

# CentOS / RHEL
sudo yum install -y git python3 python3-pip
```

### 第 2 步 · 克隆仓库

```bash
git clone https://github.com/lynnquams/-.git ~/clearchem
cd ~/clearchem
```

> 仓库约 1.2 GB（含分卷权重），国内直连 Gitee 通常 2–5 分钟。

### 第 3 步 · 跑部署脚本

```bash
bash scripts/deploy.sh --lite
```

脚本会依次做：
1. 探测 Python 版本、GPU、磁盘、外网 —— 缺什么就禁用什么，不会中断
2. 装依赖（numpy / pandas / scipy / rdkit / torch，走清华镜像）
3. 合并分卷权重并校验
4. 下载 CALiSol-23 配方数据
5. 写 `clearchem/config.json`
6. **真实调用每一层做自检**

### 第 4 步 · 确认自检通过

正常输出：

```
  ✓ 依赖与 RDKit                 torch 2.x · rdkit 2024.x · cuda 无
  ✓ 五把分子尺子                 gap(0.404) ip(0.389) ea(0.419) homo(0.326) lumo(0.280)
  ✓ 性质预测                     EC gap=8.12 eV
  ✓ 配方推荐                     LiPF6 1.17M → 10.00 mS/cm
  — 分子生成                     ether0 底座未就位
  — 知识层权重                   Qwen 底座未就位

全部通过。 标 — 的是可选项（缺底座时跳过，不影响其余功能）
```

**标 ✓ 的必须全过**，标 — 的是这一档本来就没有的功能。

### 第 5 步 · 跑起来

```bash
python3 scripts/quickstart.py
```

或者写自己的代码：

```python
from clearchem import ClearChem
cc = ClearChem(load_generator=False)          # 轻量档必须传 False

cc.predict(["C1COC(=O)O1", "COC(=O)OC"])      # 性质预测
cc.design_formulation(k_target=10.0, n=5)     # 配方推荐
```

---

## 场景 B：完整部署

**适用**：显存 ≥ 24 GB。在场景 A 基础上增加分子生成。

### 第 1–2 步 · 同场景 A

### 第 3 步 · 装 CUDA 版 torch

```bash
# 先确认 CUDA 版本
nvidia-smi | grep "CUDA Version"

# 按版本装（例：CUDA 12.1）
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

### 第 4 步 · 完整部署

```bash
cd ~/clearchem
bash scripts/deploy.sh
```

比轻量档多一步：**下载 ether0 底座（约 48 GB，走 hf-mirror 镜像）**。
这一步最久，30 分钟到 2 小时不等，支持断点续传。中断了重跑同一条命令即可。

### 第 5 步 · 自检应该多两行

```
  ✓ 分子生成                     N#CC1(O)CCCC1 gap=8.18
```

### 第 6 步 · 用起来

```python
from clearchem import ClearChem
cc = ClearChem()                                        # 默认加载生成器

r = cc.design_molecule({"gap": 8.5, "lumo": 0.5}, n=5)  # 给目标性质要分子
for x in r["results"]:
    print(x["smiles"], x["predicted"], "SA=%.2f" % x["sa"])
```

---

## 场景 C：含知识层

**适用**：显存 ≥ 60 GB（A100 80G / H100 / 双卡 A6000）。

### 一条命令，底座自动下

```bash
cd ~/clearchem && WITH_QWEN=1 bash scripts/deploy.sh
```

脚本会自动：
1. 下载 Qwen3.8-27B 底座（54 GB，走镜像，断点续传，失败自动重试 3 次）
2. 下载 ether0 底座（48 GB）
3. 下载 ChemBench 题库并装官方计分库
4. **校验底座与适配器维度是否匹配**（下错版本会当场报出来）

显存 ≥60 GB 且磁盘 ≥130 GB 时会自动触发，不需要 `WITH_QWEN=1`。

### 加载知识层

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("~/clearchem/bases/qwen", trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained("~/clearchem/bases/qwen",
                                            torch_dtype="bfloat16", device_map="auto",
                                            trust_remote_code=True)
model = PeftModel.from_pretrained(base, "~/clearchem/models/clearchem-qwen")
model.eval()
```

**注意**：ChemBench 0.6445 那个成绩是 **agent 系统**成绩，数值题接了 Python 工具执行。
纯模型（不接工具）实测 0.6316。工具的实现见 `docs/TECHNICAL.md` 第 1.3 节。

---

## 常见问题

### 权重是分卷的，合并失败怎么办

```bash
cd ~/clearchem
ls models/_parts/                      # 应该有 13 个 .part 文件
bash scripts/assemble_weights.sh       # 手动合并，幂等可重跑
```

如果报"缺分卷"，说明 clone 不完整，重新 `git pull`。

### 为什么用分卷不用 Git LFS

Gitee 免费仓库不支持 LFS（需付费企业版），且单文件有大小限制。
分卷每片 90 MB，合并脚本会校验 safetensors 头部，不匹配直接报错。

### 没有外网怎么办

打分器和配方推荐**完全离线可用**（权重在仓库里，CALiSol 数据可以从别处拷）：

```bash
bash scripts/deploy.sh --lite --skip-check   # 跳过需要联网的步骤
bash scripts/assemble_weights.sh             # 手动合并权重
```

只需要把 `data/CALiSol-23.csv` 手动放进去。

### 显存不够，加载生成器时 OOM

```python
cc = ClearChem(load_generator=False)   # 只用打分器和配方层
```

打分器是 Morgan 指纹 + MLP，**纯 CPU 也能跑**，单分子毫秒级。

### 自检某一项不过

自检会打印具体原因。常见的：

| 报错 | 原因 | 处理 |
|---|---|---|
| `缺 gap_seed17.pt` | clone 不完整 | `git pull` 后重跑合并 |
| `配方尺子未加载` | CALiSol 数据缺失 | 手动放 `data/CALiSol-23.csv` |
| `EC 的 gap 算成 X，明显不合理` | 权重损坏 | 删掉合并产物重新合并 |
| `ether0 底座未就位` | 正常（轻量档） | 需要生成功能才要处理 |

---

## 部署后的第一件事

**读 [BENCHMARKS.md](BENCHMARKS.md) 的"能力边界"一节。**

特别是配方层：跨体系外推 5 折交叉验证 R² 全部 ≤ 0.29（三折为负），
**只在 CALiSol 覆盖的 14 种锂盐 × 38 种溶剂范围内可信**。
每个接口的返回值都带 `caveat` 字段，不要忽略它。
