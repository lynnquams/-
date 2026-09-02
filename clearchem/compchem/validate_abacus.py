"""ABACUS 输入文件校验器 —— 规则全部来自 282 条真实收敛计算的统计。

为什么校验器是重点：文献里三个独立来源都指向同一件事，
改脚手架的增益是改权重的 3~4 倍。Purdue 的 LAMMPS 研究里，
把校验做成模型边写边调用的 skill，从裸生成 46% 提到 5/6 全对。

规则的取舍原则：**合法性来自官方文档与源码，典型性才来自语料**。
这两者当初被我混为一谈，结果是：
  · smearing_method 只认 gaussian，把 ABACUS 同样接受的 gauss 判成非法
  · mixing_beta 范围写成 0.3~0.4，而官方默认就是 0.8（nspin=1 时）
两处都是拿 282 条有偏样本当成了全集。用 AbacusCopilot 生成的输入交叉验证时暴露的。

另一条被否掉的规则：16 个未收敛样本里前 5 个都没设 mixing，看着像规律，
但 Fisher 精确检验 p=0.379（失败率 7.7% vs 4.5%）—— 样本太少，不成规则。
"""
import re

# 282 条真实收敛计算里 100% 出现的参数
REQUIRED = ["calculation", "basis_type", "ecutwfc", "scf_thr", "scf_nmax",
            "smearing_method", "smearing_sigma"]

# 实测取值范围（min, max, 中位）。越界不一定错，但要提示。
# 取值范围：下限来自物理合理性，上限放宽到官方文档允许的区间。
# 越界只给警告，不判错 —— 合法与否由文档定，语料只说明什么常见。
RANGES = {
    "ecutwfc":        (20, 200, 50),        # Ry，官方默认 50(pw)/100(lcao)
    "scf_thr":        (1e-10, 1e-4, 1e-7),  # 官方默认 1e-9(pw)/1e-7(lcao)
    "scf_nmax":       (20, 500, 100),
    "smearing_sigma": (0.001, 0.5, 0.015),  # Ry
    "mixing_beta":    (0.0, 1.0, 0.8),      # 官方默认 0.8(nspin=1)、0.4(nspin=2/4)
}

ENUMS = {
    "calculation":     {"scf", "nscf", "relax", "cell-relax", "md", "get_wf",
                        "get_S", "gen_bessel", "get_pchg", "test_neighbour"},
    "basis_type":      {"pw", "lcao", "lcao_in_pw"},
    # 源码 occupy.cpp 与官方文档：gauss 与 gaussian 等价，mv 与 cold 等价
    "smearing_method": {"gauss", "gaussian", "fixed", "mp", "mp2", "mp3",
                        "methfessel-paxton", "mv", "cold", "fd"},
    "mixing_type":     {"broyden", "pulay", "plain"},
}

# ks_solver 与基组绑定：平面波用 dav 系，LCAO 用对角化库。写错会直接报错退出。
KS_BY_BASIS = {
    "pw":   {"dav", "dav_subspace", "cg", "bpcg"},
    "lcao": {"genelpa", "scalapack_gvx", "elpa", "lapack", "cusolver"},
}


def parse(text):
    """ABACUS INPUT：一行一个 key value，# 之后是注释。"""
    d = {}
    for ln in text.splitlines():
        ln = ln.split("#")[0].strip()
        if not ln or ln.upper().startswith("INPUT_PARAMETERS"):
            continue
        p = ln.split(None, 1)
        if len(p) == 2:
            d[p[0].lower()] = p[1].strip()
    return d


def validate(text, stru_text=None):
    """返回 {ok, errors, warnings, parsed}。errors 非空表示几乎必然跑不起来。"""
    err, warn = [], []
    d = parse(text)

    if not text.strip().upper().startswith("INPUT_PARAMETERS"):
        err.append("首行必须是 INPUT_PARAMETERS，ABACUS 不认别的开头")

    for k in REQUIRED:
        if k not in d:
            err.append("缺必需参数 %s（282 条真实收敛计算里 100%% 都有）" % k)

    for k, allowed in ENUMS.items():
        if k in d and d[k].lower() not in allowed:
            err.append("%s=%s 不是合法值，可选 %s" % (k, d[k], "/".join(sorted(allowed))))

    for k, (lo, hi, mid) in RANGES.items():
        if k not in d:
            continue
        try:
            v = float(d[k])
        except ValueError:
            err.append("%s=%s 不是数值" % (k, d[k]))
            continue
        if not (lo <= v <= hi):
            warn.append("%s=%g 超出实测范围 %g~%g（常用 %g）" % (k, v, lo, hi, mid))

    b = d.get("basis_type", "pw").lower()
    ks = d.get("ks_solver", "").lower()
    if ks and b in KS_BY_BASIS and ks not in KS_BY_BASIS[b]:
        err.append("ks_solver=%s 与 basis_type=%s 不兼容，%s 应用 %s"
                   % (ks, b, b, "/".join(sorted(KS_BY_BASIS[b]))))

    # nscf 必须接在已有电荷密度之后
    if d.get("calculation") == "nscf" and d.get("init_chg", "").lower() != "file":
        warn.append("calculation=nscf 通常需要 init_chg=file，否则读不到已收敛的电荷密度")

    # 结构与输入的一致性：STRU 里的元素种类要对得上
    if stru_text:
        els = re.findall(r"^\s*([A-Z][a-z]?\d*)\s+[\d.]+\s+\S+\.(?:upf|UPF)",
                         stru_text, re.M)
        if not els:
            warn.append("STRU 里没解析到 ATOMIC_SPECIES 的赝势行")
        if "ntype" in d and els and int(float(d["ntype"])) != len(els):
            err.append("ntype=%s 与 STRU 里的 %d 种元素不符" % (d["ntype"], len(els)))

    return {"ok": not err, "errors": err, "warnings": warn, "parsed": d}


def demo():
    good = """INPUT_PARAMETERS
calculation              scf
basis_type               pw
ecutwfc                  60
scf_thr                  1e-7
scf_nmax                 100
smearing_method          gaussian
smearing_sigma           0.015
ks_solver                dav_subspace
"""
    r = validate(good)
    assert r["ok"], r["errors"]

    # 缺必需参数
    r = validate("INPUT_PARAMETERS\ncalculation scf\nbasis_type pw\n")
    assert not r["ok"] and any("ecutwfc" in e for e in r["errors"]), r

    # ks_solver 与基组不兼容 —— 这条会让 ABACUS 直接报错退出
    r = validate(good.replace("basis_type               pw", "basis_type               lcao"))
    assert not r["ok"] and any("不兼容" in e for e in r["errors"]), r

    # 越界只警告不报错
    r = validate(good.replace("ecutwfc                  60", "ecutwfc                  500"))
    assert r["ok"] and r["warnings"], r

    # 非法枚举值
    r = validate(good.replace("smearing_method          gaussian", "smearing_method          bogus"))
    assert not r["ok"], r

    # 回归：AbacusCopilot 实际生成的输入必须无错无警
    # 这两条曾被误判 —— gauss 判成非法、mixing_beta=0.8 判成越界
    ac = """INPUT_PARAMETERS
calculation          scf
symmetry             1
kspacing             0.14
ecutwfc              100
basis_type           lcao
ks_solver            genelpa
smearing_method      gauss
smearing_sigma       0.01
mixing_type          broyden
mixing_beta          0.8
scf_nmax             100
scf_thr              1e-07
"""
    r = validate(ac)
    assert r["ok"], "AbacusCopilot 的输出被误判为错误: %s" % r["errors"]
    assert not r["warnings"], "AbacusCopilot 的输出被误警告: %s" % r["warnings"]
    print("校验器自检全部通过（含 AbacusCopilot 输出回归）")


if __name__ == "__main__":
    demo()
