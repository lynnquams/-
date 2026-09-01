"""配对 McNemar 检验。题名有重复，改用位置索引，并用 gold 答案逐位校验题序一致。"""
import json, sys, math
from pathlib import Path

import os as _os
_R = _os.environ.get("CLEARCHEM_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

B = Path(_os.path.join(_R, "results"))

def load(n):
    d = json.load(open(B / f"cb_{n}.json"))
    rs = d["results"]
    assert len(rs) == d["n_total"], f"{n}: {len(rs)} != {d['n_total']}"
    key = [(r["name"], str(r.get("gold"))) for r in rs]
    return [bool(r.get("correct")) for r in rs], key, d["fraction_correct_official"]

def mcnemar(a, b):
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0: return n01, n10, 0, 1.0
    p = min(1.0, 2 * sum(math.comb(n, i) * 0.5 ** n for i in range(min(n01, n10) + 1)))
    return n01, n10, n, p

pairs = sys.argv[1:] or ["base:sft_v2", "base:cpt_v2", "base:sft_full", "base:cpt",
                         "base:sft", "sft_v2:sft_full", "cpt_v2:sft_v2"]
print(f"{'对比':<26}{'Δ':>9}{'仅后对':>7}{'仅前对':>7}{'不一致':>7}{'p值':>9}  判定")
print("-" * 82)
for pr in pairs:
    x, y = pr.split(":")
    try:
        a, ka, fa = load(x); b, kb, fb = load(y)
    except FileNotFoundError as e:
        print(f"{pr:<26} 缺 {Path(e.filename).name}"); continue
    assert ka == kb, f"{pr}: 题序不一致，首个不同在第 {next(i for i,(u,v) in enumerate(zip(ka,kb)) if u!=v)} 题"
    n01, n10, n, p = mcnemar(a, b)
    print(f"{x+'→'+y:<26}{fb-fa:>+9.4f}{n01:>7}{n10:>7}{n:>7}{p:>9.4f}  "
          f"{'**显著**' if p < 0.05 else '噪声内，不可区分'}")

# 最小可分辨差：McNemar 下显著需 |n01-n10| > 1.96*sqrt(n_discordant)
print()
a, _, _ = load("base"); b, _, _ = load("sft_v2")
nd = sum(1 for x, y in zip(a, b) if x != y)
mde = 1.96 * math.sqrt(nd) / len(a)
print(f"典型不一致题数 {nd} → 本基准最小可分辨差 ≈ {mde:.4f}（{mde*100:.2f} 个百分点）")
print(f"低于这个幅度的任何 ChemBench 变化都测不出来，不能当成改进。")
