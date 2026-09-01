"""电解液分子轨道能级回归测试 —— 部署后一条命令自证能力。

用电化学上确定无疑的排序做判据，而非绝对值（气相 xTB 与实验口径不可直接比）：
成膜添加剂 VC/FEC 必须比 EC 更易还原，链状碳酸酯必须更难还原。

历史：训练出来的性质尺子在这套检验上 2/6（随机是 3/6），
把 VC/FEC 判成比 EC 更难还原 —— 与它们做添加剂的全部机理相反。
所以电解液分子一律走 xTB，这个测试就是那条边界的守卫。

    python tests/test_electrolyte_orbitals.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from clearchem import qm

MOLS = {"EC": "C1COC(=O)O1", "PC": "CC1COC(=O)O1", "VC": "O=C1OC=CO1",
        "FEC": "O=C1OCC(F)O1", "DMC": "COC(=O)OC", "DEC": "CCOC(=O)OCC",
        "EMC": "CCOC(=O)OC"}


def main():
    r = qm.orbitals(list(MOLS.values()))
    bad = [k for k, v in MOLS.items() if "lumo" not in r[v]]
    assert not bad, "计算失败: %s" % bad
    L = {k: r[v]["lumo"] for k, v in MOLS.items()}
    H = {k: r[v]["homo"] for k, v in MOLS.items()}

    print("%-5s %9s %9s" % ("分子", "HOMO/eV", "LUMO/eV"))
    for k in MOLS:
        print("%-5s %9.3f %9.3f" % (k, H[k], L[k]))

    checks = [("VC  比 EC 更易还原", L["VC"] < L["EC"]),
              ("FEC 比 EC 更易还原", L["FEC"] < L["EC"]),
              ("DMC 比 EC 更难还原", L["DMC"] > L["EC"]),
              ("DEC 比 EC 更难还原", L["DEC"] > L["EC"]),
              ("EMC 比 EC 更难还原", L["EMC"] > L["EC"]),
              ("FEC 抗氧化优于 EC",  H["FEC"] < H["EC"])]
    print()
    for n, g in checks:
        print("  %s  %s" % ("✓" if g else "✗", n))
    n_ok = sum(g for _, g in checks)
    print("\n通过 %d/%d" % (n_ok, len(checks)))

    # 幅度也要对：文献上添加剂比 EC 低 0.3~0.9 eV。只判方向会漏掉"压扁"的失效模式，
    # 尺子当初就是把 1.17 eV 的真实跨度压成 0.27 eV。
    for a in ("VC", "FEC"):
        d = L["EC"] - L[a]
        assert 0.2 < d < 1.2, "%s 与 EC 的 LUMO 差 %.3f eV 超出文献合理范围" % (a, d)
        print("LUMO(EC)-LUMO(%s) = %+.3f eV  ✓ 落在文献区间" % (a, d))

    assert n_ok == len(checks), "电化学排序未全过：%d/%d" % (n_ok, len(checks))
    print("\n全部通过。")


if __name__ == "__main__":
    main()
