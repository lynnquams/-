"""扩散系数分析：判断轨迹是否进入扩散区 + 更可靠的估计。

依据 Bullerjahn/von Bülow/Hummer, JCP 153, 024116 (2020)：
  ① 对 MSD 曲线做线性拟合是"de facto standard"，但各种 ad hoc 处理不可靠
  ② 不同时刻的 MSD 高度相关，忽略相关性会低估不确定度
  ③ 必须先检验"这段轨迹到底是不是扩散区"，否则拟合的是笼中振动

我之前的做法（np.polyfit 后半段）正是①，且完全没做③ ——
10 ps 轨迹算出 D 低估 1933 倍，就是把笼中平台当成了扩散斜率。

这里实现最关键的③：用 log-log 斜率判断扩散指数 α。
  MSD ∝ t^α    α≈1 扩散区   α<1 亚扩散（笼中）   α≈2 弹道区
只有 α 落在 [0.85, 1.15] 才认为可以拟合 D。
"""
import json, os, sys
import numpy as np


def analyze(t_fs, msd, tag=""):
    """t_fs: 时间(fs)  msd: 均方位移(Å²)。返回 (D_cm2s, alpha, 可信)"""
    m = (t_fs > 0) & (msd > 0)
    t, y = t_fs[m], msd[m]
    if len(t) < 20:
        return None, None, False, "点数太少 %d" % len(t)

    # 扩散指数：对数坐标下的斜率，分段看是否收敛到 1
    lt, ly = np.log(t), np.log(y)
    n = len(t)
    seg = []
    for a, b in [(n//4, n//2), (n//2, 3*n//4), (3*n//4, n)]:
        if b - a > 5:
            seg.append(float(np.polyfit(lt[a:b], ly[a:b], 1)[0]))
    # 扫描对数间隔的滞后窗口，找 α 首次稳定在 1 附近的区间（扩散区起点）。
    # 只看最长一段会把渡越段的低 α 或尾部噪声当成判决。
    wins, alpha, fit_lo, fit_hi = [], None, None, None
    # 窗口按 msd 数组的可用滞后范围取（该数组最大滞后已是轨迹的 40%）。
    # 按轨迹长度取会只覆盖前 1/10 的滞后，全落在渡越段。
    for lo_f, hi_f in [(0.05,0.25),(0.15,0.45),(0.25,0.60),(0.40,0.80),(0.55,1.00)]:
        a_, b_ = int(lo_f*n), int(hi_f*n)
        if b_ - a_ < 6:
            continue
        k_ = float(np.polyfit(lt[a_:b_], ly[a_:b_], 1)[0])
        wins.append((lo_f, hi_f, k_))
        if alpha is None and 0.9 <= k_ <= 1.1:
            alpha, fit_lo, fit_hi = k_, a_, b_
    if alpha is None and wins:
        alpha = wins[-1][2]        # 都没进扩散区，报最长窗口的值供诊断
    seg = [w[2] for w in wins]
    if alpha is None:
        return None, None, False, "分段不足"

    ok = 0.85 <= alpha <= 1.15
    # 用后半段拟合 D，但只在判定为扩散区时才认可
    h = fit_lo if fit_lo is not None else n // 2
    hi = fit_hi if fit_hi is not None else n
    sl, _ = np.polyfit(t[h:hi], y[h:hi], 1)
    D = sl / 6 * 1e-1      # Å²/fs → cm²/s

    # 不确定度：把后半段分成 5 块各自拟合，看散布
    blocks = np.array_split(np.arange(h, hi), 5)
    Ds = [np.polyfit(t[b], y[b], 1)[0] / 6 * 1e-1 for b in blocks if len(b) > 3]
    sd = float(np.std(Ds)) if len(Ds) > 2 else float("nan")

    msg = ("扩散指数 α=%.2f（分段 %s）  %s" %
           (alpha, " ".join("%.2f" % s for s in seg),
            "✓ 在扩散区" if ok else
            ("✗ 亚扩散，仍困在溶剂笼里，轨迹不够长" if alpha < 0.85 else "✗ 弹道区")))
    return D, sd, ok, msg


if __name__ == "__main__":
    f = sys.argv[1]
    d = np.load(f)
    D, sd, ok, msg = analyze(d["t"], d["msd"])
    print(msg)
    if D is not None:
        print("D = %.3e ± %.1e cm²/s" % (D, sd))
        print("文献 1M LiPF6 EC/DMC 的 Li+ ≈ 2.5e-6 cm²/s")
        if ok:
            print("→ 可信，比值 %.2f" % (D / 2.5e-6))
        else:
            print("→ 不可信，别拿这个数算电导率")
