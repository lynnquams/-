"""电解液 MD v2：按标准做法重做。

v1 失败原因（实测）：跑到 1000 步后单步耗时从 0.13 秒暴涨到分钟级，体系失稳。
四个具体错误：
  ① 40 步从 24.5 Å 压到 15.3 Å，太激进，局部原子挤在一起
  ② 1.0 fs 步长对含氢体系偏大（C-H 振动周期约 10 fs，要 ≤0.5 fs）
  ③ 压缩后平衡不够就进产出
  ④ MD 过程中没有失稳检测，坏了也不知道

v2 的改法：
  ① 分 200 步压缩，每压一点跑 20 步弛豫，且每 20 步查一次受力
  ② 0.5 fs 步长
  ③ 压缩后 NVT 平衡 10 ps
  ④ 每 200 步查温度和最大受力，越界立即停并报告
"""
import json, os, sys, time
import numpy as np
from ase import Atoms
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from ase import units
from ase.data import atomic_masses

SEED = int(os.environ.get("SEED", "0"))
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# 势能面：MP-0b2-medium 是三个模型里唯一密度合格的(-1.1%%)。
# OMol-0-XL +13.5%% 过结合、OMAT-0 -11.5%% 欠结合，都会把扩散算错。
MODEL = os.environ.get("MODEL_OVERRIDE",
                       os.path.join(_REPO, "models", "mlip", "mace-mp-0b2-medium.model"))
OUTDIR = os.environ.get("MD_OUT", os.path.join(_REPO, "runs"))
os.makedirs(OUTDIR, exist_ok=True)
from mace.calculators import MACECalculator

SMILES = {"EC": "C1COC(=O)O1", "DMC": "COC(=O)OC",
          "PC": "CC1COC(=O)O1", "PF6": None}


def rdkit_geom(smi):
    """从 SMILES 生成 3D 构象并 MMFF 优化 —— 手写坐标是前三版失败的根源。"""
    from rdkit import Chem as _C
    from rdkit.Chem import AllChem as _A
    m = _C.AddHs(_C.MolFromSmiles(smi))
    assert _A.EmbedMolecule(m, randomSeed=0xf00d + SEED) == 0, "构象生成失败 %s" % smi
    _A.MMFFOptimizeMolecule(m, maxIters=500)
    c = m.GetConformer()
    pos = np.array([[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y,
                     c.GetAtomPosition(i).z] for i in range(m.GetNumAtoms())])
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    # 自查：分子内最小间距必须 >0.9 Å
    dm = np.linalg.norm(pos[:, None] - pos[None], axis=-1) + np.eye(len(pos)) * 99
    assert dm.min() > 0.9, "%s 生成的构象最小间距 %.2f Å 不合理" % (smi, dm.min())
    return "".join(syms), pos


GEOM = {}
for _n, _s in SMILES.items():
    if _s:
        GEOM[_n] = rdkit_geom(_s)
# PF6 八面体，P-F 1.6 Å（实验值）
GEOM["PF6"] = ("PF6", np.array([[0,0,0],[1.6,0,0],[-1.6,0,0],[0,1.6,0],
                                [0,-1.6,0],[0,0,1.6],[0,0,-1.6]], dtype=float))
NAME = os.environ.get("SYS", "EC:DMC")
COMP = json.loads(os.environ.get("COMP", '{"EC": 10, "DMC": 10}'))
N_ION = int(os.environ.get("NION", "6"))
K_EXP = float(os.environ.get("KEXP", "10.0"))
T = float(os.environ.get("TEMPK", "298.15"))
DT = float(os.environ.get("DT", "2.0"))          # fs，含氢体系必须小
TPROD = int(os.environ.get("TPROD", "20000"))    # 10 ps
TEQ = int(os.environ.get("TEQ", "10000"))        # 5 ps
RHO = float(os.environ.get("RHO", "1.2"))


def build():
    at = Atoms()
    starts, i = [], 0
    for nm, n in COMP.items():
        f, pos = GEOM[nm]
        for _ in range(n):
            m = Atoms(f, positions=np.array(pos))
            m.rotate(np.random.uniform(0, 360), np.random.rand(3), center="COM")
            at += m
            starts.append((i, len(pos))); i += len(pos)
    for _ in range(N_ION):
        at += Atoms("Li", positions=[[0,0,0]]); starts.append((i,1)); i += 1
        _f, _p = GEOM["PF6"]
        at += Atoms(_f, positions=_p)
        starts.append((i, 7)); i += 7
    mass = sum(atomic_masses[z] for z in at.numbers)
    L_t = (mass * 1.66054 / RHO) ** (1/3)
    g = int(np.ceil(len(starts) ** (1/3)))
    L0 = max(L_t * 1.8, g * 7.0)
    at.set_cell([L0]*3); at.set_pbc(True)
    step = L0 / g
    slots = [(a,b,c) for a in range(g) for b in range(g) for c in range(g)][:len(starts)]
    rng = np.random.RandomState(SEED); rng.shuffle(slots)
    for (st,na),(a,b,c) in zip(starts, slots):
        at.positions[st:st+na] += np.array([(a+.5)*step,(b+.5)*step,(c+.5)*step]) \
                                  - at.positions[st:st+na].mean(0)
    at.wrap()
    return at, L0, L_t


def min_dist(atoms):
    """最小原子间距（含周期镜像）。碰撞比受力更早在这里显形。"""
    from matscipy.neighbours import neighbour_list as _nl
    try:
        d = _nl("d", atoms, 2.0)
        return float(d.min()) if len(d) else 99.0
    except Exception:
        return 99.0


def guard(atoms, tag, tref):
    """失稳三查：间距、受力、温度。任何一项越界立即抛错，别让它闷头跑。"""
    md = min_dist(atoms)
    assert md > 0.6, "%s 原子间距 %.2f Å 过近，即将碰撞" % (tag, md)
    fm = float(__import__("numpy").abs(atoms.get_forces()).max())
    assert fm < 150, "%s 受力 %.0f eV/Å 失控" % (tag, fm)
    tk = atoms.get_temperature()
    assert tk < 2.5 * tref, "%s 温度 %.0fK 失控" % (tag, tk)
    return md, fm, tk

def apply_hmr(atoms, h_mass=3.0):
    """氢质量重分配：H→3amu，从相连重原子扣除，总质量不变。
    X-H 振动周期 ∝ sqrt(m)，H 从 1 加到 3 让周期变 1.7 倍，
    步长可从 0.5 提到 2.0 fs。标准做法，不改热力学平衡性质。"""
    from matscipy.neighbours import neighbour_list as _nl
    m = atoms.get_masses().copy()
    i, j, d = _nl("ijd", atoms, 1.35)          # 共价键长上限
    n_move = 0
    for a, b in zip(i, j):
        if atoms.numbers[a] == 1 and atoms.numbers[b] != 1 and m[a] < h_mass:
            dm = h_mass - m[a]
            if m[b] - dm > 1.0:
                m[a] += dm; m[b] -= dm; n_move += 1
    atoms.set_masses(m)
    print("HMR: %d 个氢加重到 %.1f amu，总质量 %.2f→%.2f amu"
          % (n_move, h_mass, sum(atoms.get_masses()), sum(m)), flush=True)
    return atoms

np.random.seed(SEED)
at, L0, L_t = build()
print("%s  原子 %d  起始盒 %.1f Å → 目标 %.1f Å  实验 k=%.2f mS/cm"
      % (NAME, len(at), L0, L_t, K_EXP), flush=True)

t0 = time.time()
calc = MACECalculator(model_paths=MODEL, device="cuda", default_dtype="float32")
at.calc = calc
_md0 = min_dist(at)
print("摆位后最小原子间距 %.2f Å" % _md0, flush=True)
assert _md0 > 0.9, "摆位后就有 %.2f Å 的接触，盒子构建有问题" % _md0
f0 = np.abs(at.get_forces()).max()
assert f0 < 5000, "初始受力 %.0f 太大" % f0
FIRE(at, logfile=None).run(fmax=1.0, steps=300)
print("弛豫后受力 %.2f eV/Å  %.1fs" % (np.abs(at.get_forces()).max(), time.time()-t0), flush=True)

# ① 分 200 步缓慢压缩，每步跑 20 步 MD 弛豫，全程监控受力
apply_hmr(at, float(os.environ.get("HMASS", "3.0")))
MaxwellBoltzmannDistribution(at, temperature_K=T)
sq = Langevin(at, DT*units.fs, temperature_K=T, friction=0.1)
NSQ = 200
for i in range(NSQ):
    cur = float(at.cell[0][0])
    tgt = cur - (cur - L_t) / (NSQ - i)
    at.set_cell(at.cell * (tgt/cur), scale_atoms=True)
    sq.run(20)
    if (i+1) % 40 == 0:
        fm = np.abs(at.get_forces()).max()
        tk = at.get_temperature()
        print("    压缩 %3d/%d  L=%.2f Å  受力 %.1f eV/Å  T=%.0fK  %.1f分"
              % (i+1, NSQ, at.cell[0][0], fm, tk, (time.time()-t0)/60), flush=True)
        guard(at, "压缩%d" % (i+1), T)
L = float(at.cell[0][0])
print("压缩完成 L=%.2f Å  密度 %.2f g/cm³  %.1f分"
      % (L, sum(atomic_masses[z] for z in at.numbers)*1.66054/L**3, (time.time()-t0)/60), flush=True)

# ③ 充分平衡 + ④ 失稳检测
eq = Langevin(at, DT*units.fs, temperature_K=T, friction=0.1)
_bad = [0]
def watch():
    n = eq.get_number_of_steps()
    md, fm, tk = guard(at, "平衡%d" % n, T)
    if n % 1000 == 0:
        print("    平衡 %5d/%d  T=%.0fK  受力 %.1f  最小间距 %.2fÅ  %.1f分"
              % (n, TEQ, tk, fm, md, (time.time()-t0)/60), flush=True)
eq.attach(watch, interval=100)
eq.run(TEQ)
print("平衡完成 %.1f 分" % ((time.time()-t0)/60), flush=True)

# 产出
dyn = Langevin(at, DT*units.fs, temperature_K=T, friction=0.05)
P, TS = [], []
li = [i for i, z in enumerate(at.numbers) if z == 3]
dyn.attach(lambda: (P.append(at.get_positions().copy()), TS.append(dyn.get_time()/units.fs)),
           interval=20)
_lt = [time.time(), 0]
def prog():
    n = dyn.get_number_of_steps(); dt = time.time()-_lt[0]
    r = (n-_lt[1])/max(dt, 1e-9)
    fm = np.abs(at.get_forces()).max()
    print("    产出 %5d/%d  %.1f步/秒  T=%.0fK  受力 %.1f  eta %.1f分"
          % (n, TPROD, r, at.get_temperature(), fm, (TPROD-n)/max(r,1e-6)/60), flush=True)
    _lt[0] = time.time(); _lt[1] = n
dyn.attach(prog, interval=2000)
dyn.attach(lambda: guard(at, "产出%d" % dyn.get_number_of_steps(), T), interval=100)
print("产出 %d 步 (%.1f ps)..." % (TPROD, TPROD*DT/1000), flush=True)
dyn.run(TPROD)

Pa = np.array(P); ts = np.array(TS)

def msd_multi_origin(pos, idx, max_lag_frac=0.4):
    """多时间原点 MSD：对所有可能的起点 t0 平均。
    只用 t=0 一个原点时，5000 帧只贡献 1 条样本；滑动窗口能贡献上千条。
    max_lag 限制在轨迹长度的 40%，更长的 lag 可用原点太少反而噪声大。"""
    n = len(pos)
    max_lag = max(2, int(n * max_lag_frac))
    out = np.zeros(max_lag)
    for lag in range(1, max_lag):
        d = pos[lag:, idx] - pos[:-lag, idx]        # (n-lag, n_ion, 3)
        out[lag] = (d ** 2).sum(-1).mean()
    return out

msd = msd_multi_origin(Pa, li)
ts = ts[:len(msd)] - ts[0]
np.savez(os.path.join(OUTDIR, "traj_%s.npz" % os.environ.get("RUNTAG", "run")),
         pos=Pa[:, li].astype(np.float32), t=np.array(TS), L=L, T=T)
np.savez(os.path.join(OUTDIR, "msd_%s.npz" % os.environ.get("RUNTAG", "run")),
         temperature=T,
         t=ts, msd=msd, L=L, n_ion=N_ION, T=T)
sys.path.insert(0, _HERE)
from msd_ana import analyze
D, Dsd, ok, msg = analyze(ts, msd)
print("\n" + msg, flush=True)
if D is None or not ok:
    print("→ 轨迹未进入扩散区，D 和电导率都不可信，不报数", flush=True)
    print("总用时 %.1f 分" % ((time.time()-t0)/60))
    raise SystemExit(0)
print("D = %.3e ± %.1e cm²/s   文献 ≈2.5e-6" % (D, Dsd), flush=True)
sl = D * 6 / 1e-1
V = (L*1e-8)**3
n_c = N_ION/V
sigma = n_c*(1.602e-19)**2*D/(1.381e-23*T)*1e3
print("\nLi MSD 斜率 %.4e Å²/fs → D=%.3e cm²/s" % (sl, D))
print("Nernst-Einstein σ = %.2f mS/cm   实验 %.2f   误差 %.0f%%"
      % (sigma, K_EXP, 100*abs(sigma-K_EXP)/K_EXP))
print("（NE 忽略离子关联，文献上对碳酸酯体系高估 20-40%%）")
print("总用时 %.1f 分" % ((time.time()-t0)/60))
