"""ClearChem 路由器：判断一句话该走哪个模型/引擎。

为什么需要这层：ClearChem 是两个大模型加五个引擎，选错了不会报错，
只会给出一个看起来正常的错误答案。三个已经踩过的例子：

    "算一下 VC 的 LUMO"     走尺子 → 排序是反的（电化学检验 2/6）
    "生成一个 gap=8.5 的分子" 走 ChemQwen → 它会聊设计思路，不会真生成
    "EC:DMC 和 PC 哪个电导高" 走单次 MD → 换个种子排序就翻转

所以路由不能只写在文档和工具描述里，得是代码。

设计原则：只在**有实测依据**的地方硬路由，其余给建议不拦截。
每条规则都注明它的依据是哪次实测。
"""
import re

# ── 两个大模型底座 ────────────────────────────────────────────
# 生成器 bf16 约 44 GB + 知识层 54 GB = 98 GB > 单卡 85 GB，
# 不能同时驻留。所以路由不只是"选对工具"，还决定要不要换模型（约 160 秒）。
BACKENDS = {
    "generator": {"base": "ether0", "vram_gb": 44,
                  "methods": ["design_molecule"],
                  "desc": "条件分子生成，实测条件遵循 MAE 0.109 eV"},
    "knowledge": {"base": "Qwen3.8-27B", "vram_gb": 54,
                  "methods": ["ask"],
                  "desc": "化学问答，ChemBench 0.6445（接工具）"},
}
# 小模型与引擎：常驻或秒级加载，不占大显存
LIGHT = {
    "scorers":  {"methods": ["predict", "design_formulation"], "vram_gb": 0.1,
                 "desc": "五把性质尺子 + 电导率尺子"},
    "xtb":      {"methods": ["orbitals", "screen_additive"], "vram_gb": 0,
                 "desc": "GFN2-xTB 轨道能级，纯 CPU"},
    "mace":     {"methods": ["simulate_conductivity"], "vram_gb": 4,
                 "desc": "MACE 分子动力学"},
}

# ── 意图规则 ──────────────────────────────────────────────
# (正则, 目标方法, 依据)。顺序有意义：先匹配的优先。
RULES = [
    # 「设计」后面跟什么词都算生成请求：第一版写 "设计.*分子"，
    # 「设计一个新的电解液溶剂」里没有"分子"二字，直接落到 ask 上 ——
    # ChemQwen 会聊设计思路，不会真生成，而且不报错。
    (r"生成|设计|design|新分子|候选分子|想要.*gap|目标性质|做几个|来几个|给我.*个.*(分子|溶剂|添加剂)",
     "design_molecule",
     "生成/设计类请求必须走生成器。走 ChemQwen 会得到一段设计思路而不是分子。"),

    (r"成膜|SEI|添加剂|additive|film.form|还原.*先|先.*还原",
     "screen_additive",
     "成膜判断看 LUMO 相对高低，实测 VC 比 EC 低 0.372 eV、FEC 低 0.566 eV。"),

    (r"配方|电解液.*组成|formulation|盐.*溶剂|推荐.*配比",
     "design_formulation",
     "配方推荐走电导率尺子；⚠ 只在 CALiSol 覆盖范围内可信（跨文献 R² ≤0.29）。"),

    (r"电导率|conductivity|离子传导|扩散系数|diffusion",
     "simulate_conductivity",
     "⚠ 单次 MD 不能比较两个配方：换种子 D 相差 1.84~2.20 倍，排序会翻转。"),

    # 单独的 "gap" 也要匹配：第一版只认 "带隙|band gap"，
    # 结果「这个分子的 gap 是多少」落到了 ask 上。
    (r"HOMO|LUMO|带隙|\bgap\b|轨道能级|氧化.*稳定|还原.*稳定|电化学窗口",
     "orbitals",
     "轨道能级走 xTB。电解液分子上尺子的排序是反的（电化学检验 2/6）。"),
]


def route(text, smiles=None):
    """判断一句自然语言该走哪个方法。

    返回 {method, backend, reason, needs_swap, alternatives}。
    needs_swap 为真时说明要换大模型（约 160 秒），值得先告诉用户。
    """
    t = text or ""
    hit = None
    for pat, method, why in RULES:
        if re.search(pat, t, re.I):
            hit = (method, why)
            break
    if hit is None:
        hit = ("ask", "没有匹配到具体计算意图，按化学问答处理。")

    method, why = hit

    # 分子类型再修一次。不能只在 method=="predict" 时才检查：
    # 「这个分子的 gap 是多少」会先落到 ask 上，带着 VC 也不会被改道。
    # 只要提到了电解液分子，且当前不是已经走 xTB 的方法，就改道。
    if smiles and method in ("predict", "ask"):
        # 用文件路径加载，不依赖包上下文：直接跑 router.py 时
        # "clearchem" 不是包，from clearchem.qm import 会失败。
        import importlib.util, os as _os
        _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "qm.py")
        _sp = importlib.util.spec_from_file_location("_ccqm", _p)
        _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
        _electrolyte_like = _m._electrolyte_like
        bad = {s: _electrolyte_like(s) for s in smiles}
        bad = {k: v for k, v in bad.items() if v}
        if bad:
            method = "orbitals"
            why = ("检测到电解液类分子 %s，改走 xTB —— "
                   "尺子在这些分子上电化学排序 2/6，把 VC/FEC 判反。"
                   % ", ".join("%s(%s)" % (k[:16], v) for k, v in bad.items()))

    backend = next((b for b, d in BACKENDS.items() if method in d["methods"]), None)
    if backend is None:
        backend = next((b for b, d in LIGHT.items() if method in d["methods"]), "scorers")

    return {"method": method, "backend": backend, "reason": why,
            "needs_big_model": backend in BACKENDS,
            "vram_gb": (BACKENDS.get(backend) or LIGHT.get(backend, {})).get("vram_gb", 0)}


def demo():
    cases = [
        ("帮我生成 5 个 gap 大约 8.5 的分子", "design_molecule"),
        ("设计一个新的电解液分子", "design_molecule"),
        ("设计一个新的电解液溶剂", "design_molecule"),
        ("设计几个高压添加剂", "design_molecule"),
        ("给我来几个 gap 大的分子", "design_molecule"),
        ("做几个新溶剂看看", "design_molecule"),
        ("FEC 能不能做成膜添加剂", "screen_additive"),
        ("推荐一个电导率 10 mS/cm 的配方", "design_formulation"),
        ("算一下 EC:DMC 的电导率", "simulate_conductivity"),
        ("VC 的 LUMO 是多少", "orbitals"),
        ("碳酸乙烯酯的分子式是什么", "ask"),
        ("什么是 SEI 膜", "screen_additive"),   # 含"SEI"，按成膜处理
    ]
    bad = []
    for text, want in cases:
        r = route(text)
        ok = r["method"] == want
        print("  %s %-30s → %-22s %s" % ("✓" if ok else "✗", text[:30],
                                         r["method"], r["backend"]))
        if not ok:
            bad.append((text, want, r["method"]))
    # 电解液分子即使走 predict 也要被改道
    for text, smi, want in [
        ("这个分子的 gap 是多少", ["O=C1OC=CO1"], "orbitals"),
        ("帮我算算这个的性质", ["C1COC(=O)O1"], "orbitals"),
        ("这个分子的 gap 是多少", ["c1ccccc1"], "orbitals"),   # 苯不是电解液，但问 gap 仍走 xTB
    ]:
        r = route(text, smiles=smi)
        ok = r["method"] == want
        print("  %s %-30s + %-14s → %s"
              % ("✓" if ok else "✗", text[:30], smi[0][:14], r["method"]))
        if not ok:
            bad.append((text + " " + smi[0], want, r["method"]))
    assert not bad, "路由错误：%s" % bad
    print("路由自检通过")


if __name__ == "__main__":
    demo()
