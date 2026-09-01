"""ABACUS 输入文件校验器 —— 规则全部来自 282 条真实收敛计算的统计。

为什么校验器是重点：文献里三个独立来源都指向同一件事，
改脚手架的增益是改权重的 3~4 倍。Purdue 的 LAMMPS 研究里，
把校验做成模型边写边调用的 skill，从裸生成 46% 提到 5/6 全对。

规则的取舍原则：只写证据站得住的。
曾观察到 16 个未收敛样本里前 5 个都没设 mixing，
但 Fisher 精确检验 p=0.379（失败率 7.7% vs 4.5%）—— 样本太少，不成规则，没写进来。
"""
import re

# 282 条真实收敛计算里 100% 出现的参数
REQUIRED = ["calculation", "basis_type", "ecutwfc", "scf_thr", "scf_nmax",
            "smearing_method", "smearing_sigma"]

# 实测取值范围（min, max, 中位）。越界不一定错，但要提示。
RANGES = {
    "ecutwfc":        (30, 120, 60),      # Ry
    "scf_thr":        (1e-7, 1e-5, 1e-7),
    "scf_nmax":       (60, 200, 100),
    "smearing_sigma": (0.005, 0.1, 0.015),  # Ry
    "mixing_beta":    (0.3, 0.4, 0.4),
}

ENUMS = {
    "calculation":     {"scf", "nscf", "relax", "cell-relax", "md", "get_wf",
                        "get_S", "gen_bessel", "get_pchg", "test_neighbour"},
    "basis_type":      {"pw", "lcao", "lcao_in_pw"},
    "smearing_method": {"gaussian", "fixed", "mp", "mv", "fd"},
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
    print("校验器自检全部通过")


if __name__ == "__main__":
    demo()
