# 模型权重

全部权重通过 Git LFS 存储。克隆后需执行 `git lfs pull` 才能取到实体文件。

```
models/
  clearchem-qwen/          知识层 LoRA 适配器（305 MB）
    adapter_model.safetensors
    adapter_config.json
  clearchem-gen/           分子生成器 LoRA + 条件编码器（745 MB）
    adapter_model.safetensors
    adapter_config.json
    cond_encoder.pt        性质向量 → 前缀 embedding 的 2 层 MLP
  scorers/                 六把打分器（53 MB）
    gap_seed17.pt          MAE 0.404 eV · Spearman 0.907
    homo_seed17.pt         MAE 0.326 eV · Spearman 0.840
    lumo_seed17.pt         MAE 0.280 eV · Spearman 0.951
    ip_seed17.pt           MAE 0.389 eV · Spearman 0.791（样本少，结论打折）
    ea_seed17.pt           MAE 0.419 eV · Spearman 0.801（同上）
    cond_calisol.pt        电解液电导率（跨体系外推不可信，见 docs/BENCHMARKS.md）
```

## 需要自备的底座

两个 LoRA 适配器不含底座权重，需自行获取：

| 适配器 | 底座 | 大小 |
|---|---|---|
| clearchem-qwen | Qwen3.8-27B | ~54 GB |
| clearchem-gen | ether0 | ~48 GB |

`scripts/deploy.sh` 会自动下载 ether0（走 hf-mirror 镜像）。
Qwen 底座因体积较大，建议手动准备后用 `--no-base` 部署。

## 打分器可独立使用

六把打分器是自包含的（Morgan 指纹 + MLP，不依赖任何大模型），
在纯 CPU、无网络的机器上也能跑：

```python
import torch, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

ck = torch.load("models/scorers/gap_seed17.pt", map_location="cpu", weights_only=False)
print(ck["test_mae"], ck["test_spearman"])   # 0.404  0.907
```

完整用法见 `clearchem/clearchem.py` 的 `predict()`。

## 校验

```bash
python3 -m clearchem.selfcheck
```

自检会真实调用每一层并与已知基准对照，不做"import 成功就算通过"的检查。

## 版本对照（详见 TRAINING.md）

| 目录 | 训练内容 | ChemBench | 说明 |
| --- | --- | --- | --- |
| `clearchem-qwen` | 偏好靶向 + 偏好蒸馏 + RDKit 成对 | 0.6316 纯 / **0.6445** 接工具 | **推荐**，比裸底座 +4.81pp |
| `clearchem-qwen-dpo` | 从 v1 续训 DPO，50,925 对 | 0.6352 纯 / 0.6474 接工具 | 与 v1 统计上不可区分（p=0.38 / 0.51） |
| `clearchem-gen` | 条件分子生成（ether0 底座） | 条件遵循 MAE 0.109 eV | Novelty 0.928 · Validity 0.992 |
| `scorers` | 五把性质尺子 + 电导率尺子 | MAE 0.280~0.419 | ⚠ 电解液分子上失效，走 xTB |
| `mlip` | MACE-MP-0b2 势能面 | 密度偏差 −1.1% | ⚠ 种子波动大，不能排序配方 |

**最小可分辨差 1.15pp** —— DPO 版那 +0.29pp 测不出来，对外一律写 0.6445。
