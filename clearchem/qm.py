"""GFN2-xTB 轨道能级 —— 电解液分子的还原/氧化稳定性。

为什么必须有这一层：训练出来的性质尺子在电解液分子上不可用。
实测同一套 6 条电化学排序检验：

    尺子 2/6（比随机的 3/6 还差）   xTB 6/6

尺子把碳酸酯 LUMO 的真实跨度 1.17 eV 压成 0.27 eV，
而它自身测试 MAE 就是 0.28 eV —— 要分辨的差距比工具误差还小，必然出错。
最致命的是它把 VC/FEC 判成比 EC 更难还原，方向与成膜机理完全相反。

尺子在通用小分子宽分布上 Spearman 0.951 是真的，
但那里 LUMO 跨好几个 eV；挤进碳酸酯窄带就失效。分布内可信 ≠ 处处可信。
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

HARTREE_EV = 27.211386245988
BOHR_PER_ANGSTROM = 1.8897259886


# 电解液类分子的子结构：碳酸酯、砜、腈、醚、磷酸酯。
# 尺子在这些分子上电化学排序只有 2/6（随机是 3/6），且把 VC/FEC 判成
# 比 EC 更难还原 —— 与它们做成膜添加剂的机理完全相反。
# 光在文档里写"别用"不够：predict() 照算不误，调用方不看文档就会拿到反的结论。
_ELECTROLYTE_SMARTS = [
    # 环状碳酸酯用通配的环键，不能写死饱和环：
    # 第一版写 "O=C1OCCO1"，VC（O=C1OC=CO1）环上是双键，直接漏掉 ——
    # 而 VC 正是尺子判反的那个分子，也是最经典的成膜添加剂。
    ("环状碳酸酯", "[#8]=[#6]1[#8][#6][#6][#8]1"),
    ("链状碳酸酯", "[#6][#8][#6](=[#8])[#8][#6]"),
    ("砜/亚砜",   "[#16X4](=O)(=O)"),
    ("腈",       "[CX2]#[NX1]"),
    ("醚",       "[OD2]([#6])[#6]"),
    ("磷酸酯",    "[PX4](=O)(O)(O)O"),
]
_ELEC_PATTS = None


def _electrolyte_like(smi):
    """返回命中的电解液子结构名，没命中返回 None。"""
    global _ELEC_PATTS
    if _ELEC_PATTS is None:
        _ELEC_PATTS = [(n, Chem.MolFromSmarts(q)) for n, q in _ELECTROLYTE_SMARTS]
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumHeavyAtoms() > 30:
        return None
    for name, patt in _ELEC_PATTS:
        if patt is not None and m.HasSubstructMatch(patt):
            return name
    return None

def _geometry(smi, seed=0xf00d):
    """RDKit ETKDG + MMFF。手写坐标是 MD 前三版全部失败的根源，这里不重蹈。"""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError("SMILES 无法解析: %s" % smi)
    m = Chem.AddHs(m)
    if AllChem.EmbedMolecule(m, randomSeed=seed) != 0:
        raise ValueError("构象生成失败: %s" % smi)
    AllChem.MMFFOptimizeMolecule(m, maxIters=2000)
    c = m.GetConformer()
    Z = np.array([a.GetAtomicNum() for a in m.GetAtoms()])
    R = np.array([list(c.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    return Z, R * BOHR_PER_ANGSTROM


def orbitals(smiles, seed=0xf00d):
    """返回 {smiles: {homo, lumo, gap, dipole}}，单位 eV / Debye。约 1 秒/分子。"""
    from tblite.interface import Calculator
    if isinstance(smiles, str):
        smiles = [smiles]
    out = {}
    for s in smiles:
        try:
            Z, R = _geometry(s, seed)
            calc = Calculator("GFN2-xTB", Z, R)
            calc.set("verbosity", 0)
            d = calc.singlepoint().dict()
            e = np.asarray(d["orbital-energies"]) * HARTREE_EV
            occ = np.asarray(d["orbital-occupations"])
            homo, lumo = float(e[occ > 0.5].max()), float(e[occ <= 0.5].min())
            out[s] = {"homo": round(homo, 3), "lumo": round(lumo, 3),
                      "gap": round(lumo - homo, 3),
                      "dipole": round(float(np.linalg.norm(d["dipole"])) * 2.5417464, 3),
                      "n_atoms": len(Z)}
        except Exception as exc:
            out[s] = {"error": str(exc)[:200]}
    return out


def screen_additive(smiles, reference="C1COC(=O)O1"):
    """判断候选能否做成膜添加剂：需比参照溶剂(默认 EC)更易还原，即 LUMO 更低。

    实测参照点：VC 比 EC 低 0.372 eV、FEC 低 0.566 eV（文献 0.3~0.9，吻合）。
    """
    r = orbitals(list({*([smiles] if isinstance(smiles, str) else smiles), reference}))
    ref = r.get(reference, {})
    if "lumo" not in ref:
        return {"error": "参照分子计算失败", "detail": ref}
    res = {}
    for s in ([smiles] if isinstance(smiles, str) else smiles):
        d = r.get(s, {})
        if "lumo" not in d:
            res[s] = d
            continue
        dl = ref["lumo"] - d["lumo"]
        res[s] = {**d, "lumo_below_ref_eV": round(dl, 3),
                  "sei_forming": bool(dl > 0.2),
                  "caveat": ("GFN2-xTB 气相单构象。溶剂化会整体平移能级，"
                             "故只用于同类分子间的相对比较，不作绝对电位。")}
    return {"reference": reference, "reference_lumo": ref["lumo"], "candidates": res}
