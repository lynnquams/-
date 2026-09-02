"""持续计算难算例 —— 产出「输入 + 执行结果 + 报错」配对语料。

文献里说化学领域没有把输入与执行结果配对的数据集
（两个可下载的都是 CFD 的 NL2FOAM / FoamGPT）。跑不通的样本尤其值钱：
它带着报错信息，是纠错轨迹的唯一来源。

难 = 我们数据集里稀缺且算得慢的那些：
    自旋轨道耦合 lspinorb · DFT+U · 杂化泛函 · 无轨道 DFT
    分子动力学 · 全弛豫 cell-relax · 电导 cal_cond · 自旋极化 nspin

资源约束（实测，不是读 nproc）：
    westc 的 nproc 报 48，cgroup 真实配额 1800000/100000 = 18 vCPU
    按 nproc 起进程会超订 2.7 倍，把机器 thrash 死
    磁盘只剩 3.9G —— 每跑完必须清输出，否则几十个算例就撑爆
"""
import json, os, re, shutil, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_abacus import run

SRC = Path(os.environ.get("ABACUS_SRC",
                          "/root/autodl-tmp/mol-discover/abacus-src"))
OUT = Path(os.environ.get("QUEUE_OUT", "/root/hard_runs.jsonl"))
# 18 vCPU 真实配额，留 2 核给系统；每个作业 2 进程 × 2 线程 = 4 核
CONCURRENT = int(os.environ.get("CONCURRENT", "4"))
TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "2400"))

# 越靠前越难、越稀缺
HARD_FEATURES = [
    ("lspinorb", 10), ("dft_plus_u", 9), ("exx_hybrid_alpha", 9),
    ("dft_functional", 7), ("of_kinetic", 7), ("cal_cond", 7),
    ("deepks_scf", 6), ("sc_mag_switch", 6), ("berry_phase", 6),
    ("esolver_type", 5), ("nspin", 4), ("vdw_method", 4),
    ("implicit_solvation", 4), ("lr_nstates", 4),
]
HARD_CALC = {"md": 8, "cell-relax": 7, "relax": 5, "nscf": 3, "scf": 0}


def kv(text):
    d = {}
    for ln in text.splitlines():
        ln = ln.split("#")[0].strip()
        p = ln.split(None, 1)
        if len(p) == 2:
            d[p[0].lower()] = p[1].strip()
    return d


def score(inp):
    k = kv(inp)
    s = HARD_CALC.get(k.get("calculation", "scf"), 0)
    for f, w in HARD_FEATURES:
        if f in k:
            s += w
    if k.get("basis_type") == "lcao":
        s += 2
    return s


def build_queue():
    q = []
    for f in SRC.rglob("INPUT"):
        d = f.parent
        if not (d / "STRU").exists():
            continue
        ref = d / "result.ref"
        if not (ref.exists() and ref.read_text(errors="ignore").strip()):
            continue          # 没有参考值就判不了对错，跳过
        inp = f.read_text(errors="ignore")
        q.append({"dir": str(d), "score": score(inp),
                  "calc": kv(inp).get("calculation", "scf")})
    q.sort(key=lambda x: -x["score"])
    return q


def do_one(item):
    d = Path(item["dir"])
    ref_txt = (d / "result.ref").read_text(errors="ignore")
    m = re.search(r"etotref\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)", ref_txt)
    ref = float(m.group(1)) if m else None
    wd = "/tmp/hq_%s" % d.name[:40]
    try:
        r = run((d / "INPUT").read_text(errors="ignore"),
                (d / "STRU").read_text(errors="ignore"),
                (d / "KPT").read_text(errors="ignore") if (d / "KPT").exists() else None,
                workdir=wd, nproc=2, timeout=TIMEOUT)
    except Exception as e:
        r = {"ok": False, "error": "执行器异常: %s" % str(e)[:200]}
    rec = {"case": d.name, "dir": str(d), "hardness": item["score"],
           "calculation": item["calc"], "reference_etot": ref,
           "input": (d / "INPUT").read_text(errors="ignore")[:4000],
           "stru": (d / "STRU").read_text(errors="ignore")[:3000],
           **{k: v for k, v in r.items() if k != "workdir"}}
    if ref is not None and r.get("etot_eV") is not None:
        rec["abs_err_eV"] = abs(r["etot_eV"] - ref)
        rec["matches_reference"] = rec["abs_err_eV"] < 1e-4
    # 磁盘只剩几个 G，跑完立刻清
    shutil.rmtree(wd, ignore_errors=True)
    return rec


def main():
    done = set()
    if OUT.exists():
        for ln in OUT.read_text().splitlines():
            try: done.add(json.loads(ln)["dir"])
            except Exception: pass
    q = [x for x in build_queue() if x["dir"] not in done]
    print("队列 %d 个难算例（已完成 %d 个，跳过）  并发 %d"
          % (len(q), len(done), CONCURRENT), flush=True)
    print("最难的前 5：%s\n" % ", ".join(
        "%s(%d)" % (Path(x["dir"]).name[:26], x["score"]) for x in q[:5]), flush=True)

    ok = fail = match = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=CONCURRENT) as ex, open(OUT, "a") as f:
        futs = {ex.submit(do_one, x): x for x in q}
        for i, fu in enumerate(as_completed(futs), 1):
            rec = fu.result()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            ok += rec.get("ok", False); fail += not rec.get("ok", False)
            match += rec.get("matches_reference", False)
            if i % 5 == 0 or i == len(q):
                print("  %d/%d  跑通 %d  失败 %d  数值对上 %d  %.1f分"
                      % (i, len(q), ok, fail, match, (time.time() - t0) / 60), flush=True)
    print("\n完成：跑通 %d  失败 %d  数值对上 %d  用时 %.1f 分"
          % (ok, fail, match, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
