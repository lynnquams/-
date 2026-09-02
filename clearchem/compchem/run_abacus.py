"""ABACUS 执行器 —— 把输入文件真跑一遍，返回可判对错的数值。

两个坑是实测踩出来的，固化在这里：

① 必须用与二进制配套的 MPI。
   westc 上系统是 Open MPI 4.1.2，而 ABACUS 链接的是工具链里的 MPICH 4.3.2。
   用错的那个去起，进程会挂在 MPI_Init 上 —— 不报错、不建输出目录、
   连 timeout 的 SIGTERM 都不响应，看起来像"MPI 坏了"。

② 进程数不能写死。
   小算例 k 点少，4 进程会报 "nks == 0, some processor have no k point"。
   按 k 点数自适应，宁可少开。

③ nscf 依赖前一步 scf 的电荷密度。
   直接跑会报 "Can't open file ...CHARGE-DENSITY.restart"。
   实测 433 个算例里有 15 个栽在这上面 —— 执行器要能自己先补一步 scf。

④ 缺轨道文件是最大单一失败原因（37/150），但**补库不能混进同一个目录**。
   实测把 SG15 全套摊平合并后：新跑通 12 个，却让 28 个原本数值精确
   对上的算例算出不同的数（-1889.1753 → -1889.1982）—— 净亏 27 个。
   算例自带的赝势被同名或先匹配的覆盖了。
   正确做法：原路径优先，找不到再回退到补充库。
"""
import os, re, shutil, subprocess, time
from pathlib import Path

TOOLCHAIN = os.environ.get(
    "ABACUS_TOOLCHAIN",
    "/root/autodl-tmp/mol-discover/envs/abacus-mpi-toolchain-cpu")
BINARY = os.environ.get(
    "ABACUS_BIN",
    "/root/autodl-tmp/mol-discover/build-abacus-gpu/abacus")
# 主库：算例原本用的那套，优先
PP_DIR = os.environ.get(
    "ABACUS_PP", "/root/autodl-tmp/mol-discover/abacus-src/tests/PP_ORB")
# 补充库：主库里找不到某个文件时才用，绝不覆盖主库
PP_FALLBACK = os.environ.get("ABACUS_PP_FALLBACK", "/root/pp_flat")


def _nproc_for(kpt_text, requested):
    """k 点少时降进程数，否则某些 rank 分不到 k 点直接报错退出。"""
    if not kpt_text:
        return min(requested, 2)
    m = re.search(r"^\s*(\d+)\s+(\d+)\s+(\d+)", kpt_text, re.M)
    if not m:
        return min(requested, 2)
    nk = int(m.group(1)) * int(m.group(2)) * int(m.group(3))
    return max(1, min(requested, nk))


def _prepare_scf_for_nscf(txt):
    """nscf 需要已有的电荷密度。把同一份输入改成 scf 先跑一遍。"""
    t = re.sub(r"^\s*calculation\s+\S+", "calculation scf", txt, flags=re.M)
    t = re.sub(r"^\s*init_chg\s+\S+", "init_chg atomic", t, flags=re.M)
    if "out_chg" not in t:
        t = t.rstrip() + "\nout_chg 1\n"
    else:
        t = re.sub(r"^\s*out_chg\s+\S+", "out_chg 1", t, flags=re.M)
    return t


def run(input_text, stru_text, kpt_text=None, workdir=None,
        nproc=4, timeout=1800, pp_dir=None, auto_scf=True, gpu=None):
    """跑一个 ABACUS 计算。返回 dict：etot_eV / converged / seconds / error。

    auto_scf: calculation=nscf 时先自动跑一遍 scf 生成电荷密度。
    """
    w = Path(workdir or ("/tmp/abacus_%d" % int(time.time() * 1000)))
    if w.exists():
        shutil.rmtree(w)
    w.mkdir(parents=True)

    pp = pp_dir or PP_DIR
    # 主库优先：把主库整个软链进工作目录，缺的文件才从补充库补。
    # 直接把补充库合并进主库会覆盖算例自带的赝势，实测让 28 个原本
    # 数值精确对上的算例算出不同的数。
    link = w / "pp"
    link.mkdir()
    for src in (Path(pp), Path(PP_FALLBACK)):
        if not src.exists():
            continue
        for f in src.iterdir():
            if f.is_file() and not (link / f.name).exists():   # 先到先得 = 主库优先
                try: (link / f.name).symlink_to(f)
                except OSError: pass
    pp = str(link)

    txt = re.sub(r"^\s*pseudo_dir.*$", "pseudo_dir %s" % pp, input_text, flags=re.M)
    txt = re.sub(r"^\s*orbital_dir.*$", "orbital_dir %s" % pp, txt, flags=re.M)
    if "pseudo_dir" not in txt:
        txt = txt.rstrip() + "\npseudo_dir %s\norbital_dir %s\n" % (pp, pp)

    (w / "INPUT").write_text(txt)
    (w / "STRU").write_text(stru_text)
    if kpt_text:
        (w / "KPT").write_text(kpt_text)

    n = _nproc_for(kpt_text, nproc)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "%s/lib:%s" % (TOOLCHAIN, env.get("LD_LIBRARY_PATH", ""))
    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "4")
    # 指定 GPU 必须走子进程环境。在线程里改 os.environ 会被其他线程覆盖 ——
    # 实测三个并发全挤到同一张卡上，另两张 0 MiB 空转。
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    cmd = ["%s/bin/mpirun" % TOOLCHAIN, "-n", str(n), BINARY]
    t0 = time.time()
    pre = None

    # nscf 先补一步 scf，否则读不到电荷密度（实测 15/150 的失败原因）
    if auto_scf and re.search(r"^\s*calculation\s+nscf", txt, re.M | re.I):
        (w / "INPUT").write_text(_prepare_scf_for_nscf(txt))
        try:
            subprocess.run(cmd, cwd=w, env=env, capture_output=True,
                           text=True, timeout=max(300, timeout // 3))
            pre = "已自动补跑 scf"
        except subprocess.TimeoutExpired:
            pre = "补跑 scf 超时"
        (w / "INPUT").write_text(txt)

    try:
        p = subprocess.run(cmd, cwd=w, env=env, capture_output=True, text=True,
                           timeout=timeout)
        rc, out = p.returncode, p.stdout
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "超时 %ds" % timeout, "seconds": timeout,
                "nproc": n, "workdir": str(w), "prestep": pre}

    logs = list(w.glob("OUT*/running_*.log"))
    log = logs[0].read_text(errors="ignore") if logs else ""
    m = re.search(r"!FINAL_ETOT_IS\s+([-\d.eE+]+)", log)
    warn = (w / "OUT.autotest" / "warning.log")
    warns = warn.read_text(errors="ignore").strip() if warn.exists() else ""

    return {"ok": m is not None,
            "etot_eV": float(m.group(1)) if m else None,
            "converged": m is not None,
            "seconds": round(time.time() - t0, 1),
            "nproc": n, "returncode": rc, "workdir": str(w),
            "warnings": warns[:600] or None, "prestep": pre,
            "error": None if m else (out[-600:] or log[-600:] or "无输出"),
            "log_tail": log[-800:] if not m else None}


def demo():
    """回归：官方算例必须复现出参考总能。"""
    src = Path("/root/autodl-tmp/mol-discover/abacus-src/tests/integrate/101_PW_15_pseudopots")
    if not src.exists():
        print("跳过：不在 westc 上"); return
    ref = float(re.search(r"etotref\s+([-\d.]+)",
                          (src / "result.ref").read_text()).group(1))
    r = run((src / "INPUT").read_text(), (src / "STRU").read_text(),
            (src / "KPT").read_text() if (src / "KPT").exists() else None,
            workdir="/tmp/abacus_regress", nproc=4, timeout=900)
    assert r["ok"], "跑失败: %s" % r["error"]
    d = abs(r["etot_eV"] - ref)
    print("算出 %.10f   参考 %.10f   差 %.2e eV   %d 进程 %.1fs"
          % (r["etot_eV"], ref, d, r["nproc"], r["seconds"]))
    assert d < 1e-4, "总能对不上，差 %.2e eV" % d
    print("执行器回归通过")


if __name__ == "__main__":
    demo()
