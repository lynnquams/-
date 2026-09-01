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
| clearchem-qwen | Qwen3.5-27B | ~54 GB |
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
