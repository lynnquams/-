#!/usr/bin/env python3
"""最小可运行示例。部署后直接跑：python3 scripts/quickstart.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clearchem import ClearChem

# 无 GPU 或只想用打分器时传 load_generator=False
cc = ClearChem(load_generator=os.environ.get("WITH_GEN", "0") == "1")

print("\n【性质预测】常见电解液溶剂")
for smi, name in [("C1COC(=O)O1", "EC 碳酸乙烯酯"),
                  ("COC(=O)OC", "DMC 碳酸二甲酯"),
                  ("CCOC(=O)OC", "EMC 碳酸甲乙酯"),
                  ("CC1COC(=O)O1", "PC 碳酸丙烯酯")]:
    p = cc.predict([smi]).get(smi, {})
    print("  %-16s %s" % (name, "  ".join("%s=%.2f" % (k, v) for k, v in p.items())))

print("\n【配方推荐】目标电导率 10 mS/cm @ 25°C")
f = cc.design_formulation(k_target=10.0, T=298.15, n=5)
for x in f["results"]:
    sol = " : ".join("%s %.0f%%" % (k, v * 100) for k, v in x["solvents"].items())
    print("  %-9s %.2f M  %-34s → %.2f mS/cm"
          % (x["salt"][:9], x["concentration"], sol[:34], x["predicted_k"]))
print("  " + f["caveat"])

if cc.gen:
    print("\n【分子设计】目标 gap 8.5 eV、lumo 0.5 eV")
    r = cc.design_molecule({"gap": 8.5, "lumo": 0.5}, n=5)
    for x in r["results"]:
        print("  %-42s SA %-5s %s" % (x["smiles"][:42], x["sa"],
              "  ".join("%s=%.2f" % (k, v) for k, v in x["predicted"].items())))
    print("  " + r["caveat"])
else:
    print("\n【分子设计】未加载生成器（需 GPU + ether0 底座）")
    print("  设 WITH_GEN=1 并确保底座就位后重跑")
