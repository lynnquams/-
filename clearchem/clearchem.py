"""ClearChem 集成层：把六个部件串成一条可调用的链路。

在此之前部件是散的：生成、五把分子尺子、配方推荐、电导率尺子各跑各的，
分布在三台机器上，从没端到端跑通过。

链路：
    目标性质 → 生成候选分子 → 五性质打分 → 排序筛选
                              ↓
    目标电导率 → 配方推荐（盐+溶剂+浓度）→ 电导率预测

已知边界（在返回值里明确标注，不许隐瞒）：
    分子层  gap 尺子 MAE 0.404，条件遵循实测 MAE 0.109（六点偏差 ±0.03）
            ip/ea 尺子样本仅1.6万、Spearman 0.79，结论要打折
            ⚠ 尺子在电解液分子上不可用：电化学排序检验 2/6（随机是 3/6），
              且把 VC/FEC 判成比 EC 更难还原，与成膜机理相反。
              根因是碳酸酯 LUMO 真实跨度 1.17 eV 被压成 0.27 eV，
              而尺子自身 MAE 就是 0.28 eV。电解液分子一律走 qm.orbitals()。
    配方层  电导率尺子跨文献外推 5 折 R² 全部 ≤0.29（三折为负）
            → 分布内插值可信，新体系不可信
    MD 层   simulate_conductivity() 走 MACE-MP-0b2 分子动力学，不依赖实验数据，
            因而能算尺子外推不了的新体系。代价是每个配方约 3.4 小时（1 ns，单卡）。
            实测 1M LiPF6 EC:DMC：D=6.8e-07 cm²/s（文献 1.5e-6~3e-6）、电导 0.71×实验值。
            ⚠ 排序分辨力已实测且不合格：换随机种子重跑，D 相差 1.84 倍，
            大于 EC:DMC 与 PC 之间 1.72 倍的真实差距 —— 单次运行不能用于配方比较。
"""
import json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer
except Exception:
    sascorer = None

PROPS = ["ip", "ea", "gap", "homo", "lumo"]
IDX = {p: i for i, p in enumerate(PROPS)}
NORM = {"ip": (7.0, 2.0), "ea": (0.8, 1.0), "gap": (7.37, 1.88),
        "homo": (-8.5, 1.5), "lumo": (0.9, 1.0)}
# 尺子可信度：实测测试集指标，低于门槛的结论要标注打折
SCORER_TRUST = {"lumo": ("优", 0.280, 0.951), "homo": ("可用", 0.326, 0.840),
                "gap": ("可用", 0.404, 0.907), "ip": ("打折", 0.389, 0.791),
                "ea": ("打折", 0.419, 0.801)}
_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = {}
_cfg_file = os.path.join(_HERE, "config.json")
if os.path.exists(_cfg_file):
    _CFG = json.load(open(_cfg_file))
ROOT = _CFG.get("root") or os.environ.get("CLEARCHEM_ROOT") or os.path.dirname(_HERE)
SCORER_DIR = _CFG.get("scorers") or os.path.join(ROOT, "models", "scorers")
GEN_ADAPTER = (_CFG.get("adapters", {}).get("generator")
               or os.path.join(ROOT, "models", "clearchem-gen"))
GEN_BASE = (_CFG.get("bases", {}).get("ether0")
            or os.path.join(ROOT, "bases", "ether0"))
CALISOL = _CFG.get("calisol") or os.path.join(ROOT, "data", "CALiSol-23.csv")


def _mlp(din):
    return nn.Sequential(nn.Linear(din, 1024), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.1),
                         nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 1))


def _feat_raw(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32)
    d = np.array([Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.TPSA(m),
                  Descriptors.NumHDonors(m), Descriptors.NumHAcceptors(m),
                  Descriptors.NumRotatableBonds(m), Descriptors.RingCount(m),
                  rdMolDescriptors.CalcNumAromaticRings(m), Descriptors.FractionCSP3(m),
                  Descriptors.HeavyAtomCount(m), Descriptors.NumValenceElectrons(m),
                  Descriptors.MaxPartialCharge(m) or 0, Descriptors.MinPartialCharge(m) or 0],
                 dtype=np.float32)
    return fp, np.nan_to_num(d, nan=0., posinf=0., neginf=0.)


class ClearChem:
    def __init__(self, device="cuda", load_generator=True):
        self.dev = device
        self.scorers = {}
        for p in PROPS:
            f = os.path.join(SCORER_DIR, "%s_seed17.pt" % p)
            if not os.path.exists(f):
                continue
            ck = torch.load(f, map_location="cpu", weights_only=False)
            n = _mlp(2061); n.load_state_dict(ck["state_dict"]); n.eval()
            self.scorers[p] = (n, ck["mu"], ck["sd"], ck["dmu"], ck["dsd"], ck["test_mae"])
        assert len(self.scorers) >= 3, "分子尺子不足 %d" % len(self.scorers)

        cf = os.path.join(SCORER_DIR, "cond_calisol.pt")
        self.cond = None
        if os.path.exists(cf):
            ck = torch.load(cf, map_location="cpu", weights_only=False)
            din = len(ck["solvents"]) + len(ck["salts"]) + len(ck["ratios"]) + 3
            n = nn.Sequential(nn.Linear(din, 512), nn.GELU(), nn.Dropout(0.15),
                              nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.15),
                              nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
            n.load_state_dict(ck["state_dict"]); n.eval()
            self.cond = (n, ck)

        self.gen = None
        if load_generator:
            self._load_gen()
        print("ClearChem 就绪：分子尺子 %d 把 · 配方尺子 %s · 生成器 %s"
              % (len(self.scorers), "有" if self.cond else "无",
                 "已载" if self.gen else "未载"), flush=True)

    def _load_gen(self):
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        SNAP = GEN_BASE
        ADP = GEN_ADAPTER
        if not os.path.exists(ADP):
            return
        tok = AutoTokenizer.from_pretrained(SNAP, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        base = AutoModelForCausalLM.from_pretrained(SNAP, device_map={"": 0},
                                                    trust_remote_code=True)
        model = PeftModel.from_pretrained(base, ADP); model.eval()
        hid = model.get_input_embeddings().weight.shape[1]

        class CondEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(10, 512), nn.GELU(),
                                         nn.Linear(512, 8 * hid))
            def forward(self, x):
                return self.net(x)

        enc = CondEncoder()
        enc.load_state_dict(torch.load(os.path.join(ADP, "cond_encoder.pt"),
                                       map_location="cpu"))
        self.gen = (tok, model, enc.to(0).eval())

    # ---------- 分子层 ----------
    def predict(self, smis, props=None):
        """预测分子性质。返回 {smiles: {prop: value}}，附尺子可信度。"""
        props = props or list(self.scorers)
        feats, keep = [], []
        for i, s in enumerate(smis):
            r = _feat_raw(s)
            if r is not None:
                feats.append(r); keep.append(i)
        if not feats:
            return {}
        out = {smis[i]: {} for i in keep}
        for p in props:
            if p not in self.scorers:
                continue
            net, mu, sd, dmu, dsd, _ = self.scorers[p]
            X = np.stack([np.concatenate([f[0], (f[1] - dmu) / dsd]) for f in feats])
            with torch.no_grad():
                v = net(torch.tensor(X.astype(np.float32))).squeeze(-1).numpy() * sd + mu
            for i, x in zip(keep, v):
                out[smis[i]][p] = round(float(x), 3)
        return out

    def design_molecule(self, targets, n=10, k=None, temp=1.4):
        """给目标性质生成分子。K 按条件数和目标稀有度自适应（实测最优）。"""
        assert self.gen is not None, "生成器未加载"
        assert targets, "至少指定一个目标性质"
        tok, model, enc = self.gen
        if k is None:
            base_k = {1: 8, 2: 16, 3: 32}.get(len(targets), 32)
            g = targets.get("gap")
            rar = 1.0 if g is None or g <= 8.5 else (1.25 if g <= 10 else
                  (1.75 if g <= 11.5 else 2.5))
            k = int(round(base_k * rar))
        txt = "、".join("%s≈%.2f" % (a, b) for a, b in targets.items())
        prompts = ["设计一个电解液相关分子。目标性质：%s。" % txt] * (n * k)
        smis = []
        for i in range(0, len(prompts), 8):
            b = prompts[i:i+8]
            e = tok(b, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
            with torch.no_grad():
                cond = torch.zeros(len(b), 5, device=0); cm = torch.zeros(len(b), 5, device=0)
                for a_, v_ in targets.items():
                    mu_, sd_ = NORM[a_]
                    cond[:, IDX[a_]] = (v_ - mu_) / sd_; cm[:, IDX[a_]] = 1
                pe = enc(torch.cat([cond, cm], -1)).view(len(b), 8, -1)
                te = model.get_input_embeddings()(e["input_ids"])
                gg = model.generate(inputs_embeds=torch.cat([pe.to(te.dtype), te], 1),
                    attention_mask=torch.cat([torch.ones(len(b), 8, device=0,
                        dtype=e["attention_mask"].dtype), e["attention_mask"]], 1),
                    max_new_tokens=192, do_sample=True, temperature=temp, top_p=0.95,
                    pad_token_id=tok.pad_token_id)
                for x in gg:
                    t = tok.decode(x, skip_special_tokens=True).strip()
                    try:
                        dd = json.loads(t[t.index("{"):t.rindex("}")+1])
                        sm = dd["components"][0].get("smiles")
                        if sm and Chem.MolFromSmiles(sm):
                            smis.append(sm)
                    except Exception:
                        pass
        uniq = sorted(set(smis))
        if not uniq:
            return {"error": "未生成合法分子", "n_raw": len(prompts)}
        pred = self.predict(uniq, list(targets))
        out = []
        for s in uniq:
            if any(p not in pred.get(s, {}) for p in targets if p in self.scorers):
                continue
            err = sum(abs(pred[s][p] - v) / NORM[p][1]
                      for p, v in targets.items() if p in self.scorers)
            sa = float(sascorer.calculateScore(Chem.MolFromSmiles(s))) if sascorer else None
            out.append({"smiles": s, "score": round(err, 4), "predicted": pred[s],
                        "sa": round(sa, 2) if sa else None})
        out.sort(key=lambda x: x["score"])
        return {"targets": targets, "n_generated": len(smis), "n_unique": len(uniq),
                "k": k, "results": out[:n],
                "scorer_trust": {p: SCORER_TRUST[p] for p in targets if p in SCORER_TRUST},
                "caveat": "gap 尺子 MAE 0.404；ip/ea 尺子 Spearman 0.79，结论打折"}

    # ---------- 配方层 ----------
    def orbitals(self, smiles):
        """电解液分子的 HOMO/LUMO/gap —— 走 GFN2-xTB，不走尺子。

        尺子在电解液窄带内失效（排序 2/6），xTB 同一套检验 6/6 且数值合文献。
        约 1 秒/分子，不需要 GPU。
        """
        from clearchem import qm
        return qm.orbitals(smiles)

    def screen_additive(self, smiles, reference="C1COC(=O)O1"):
        """筛成膜添加剂：LUMO 比参照溶剂低多少。实测 VC 低 0.372、FEC 低 0.566 eV。"""
        from clearchem import qm
        return qm.screen_additive(smiles, reference)

    def simulate_conductivity(self, comp, rho=1.20, n_ion=6, ps=1000,
                              seed=0, k_exp=None, tag=None):
        """用分子动力学算电导率 —— 打分器外推不了的体系走这条路。

        comp   溶剂组成，如 {"EC": 10, "DMC": 10}；支持的分子见 md/run_md.py 的 SMILES
        rho    目标密度 g/cm³（体积由此定，别用默认值套新体系）
        ps     产出轨迹长度，1000 ps 约 3.4 小时/单卡

        返回值里 trustworthy=False 时 sigma 不可用：说明轨迹没进扩散区，
        MSD 还是亚扩散的，此时算出的 D 没有物理意义。
        """
        import subprocess, tempfile
        md = os.path.join(_HERE, "md", "run_md.py")
        if not os.path.exists(md):
            return {"error": "MD 组件未安装", "path": md}
        tag = tag or ("md%d" % int(time.time()))
        env = dict(os.environ,
                   COMP=json.dumps(comp), RHO=str(rho), NION=str(n_ion),
                   TPROD=str(int(ps * 1000 / 2)),   # DT=2 fs
                   TEQ="25000", SEED=str(seed), RUNTAG=tag,
                   SYS=":".join(comp), KEXP=str(k_exp if k_exp else 0.0),
                   MD_OUT=os.path.join(ROOT, "runs"))
        t0 = time.time()
        r = subprocess.run([sys.executable, md], env=env,
                           capture_output=True, text=True)
        out = r.stdout
        got = {"tag": tag, "minutes": round((time.time() - t0) / 60, 1),
               "composition": comp, "trajectory_ps": ps, "log": out[-2000:]}
        if r.returncode != 0:
            got["error"] = r.stderr[-800:]
            return got
        # 未进扩散区时脚本自身会拒绝报数，这里如实透传
        got["trustworthy"] = "不报数" not in out
        for key, pat in (("D_cm2_s", "D = "), ("sigma_mS_cm", "σ = ")):
            for ln in out.splitlines():
                if pat in ln:
                    try:
                        got[key] = float(ln.split(pat)[1].split()[0])
                    except (ValueError, IndexError):
                        pass
        got["caveat"] = ("⚠ 不可用于配方排序。同体系换随机种子实测 D 相差 1.84 倍"
                         "（6.79e-07 vs 3.70e-07 cm²/s），而 EC:DMC 与 PC 的实验电导只差 "
                         "1.72 倍 —— 自身波动大于要分辨的差距。单次结果只能读数量级。"
                         "要能排序需每配方 5~8 个种子或把离子数提到 24+，代价 17~27 小时/配方。"
                         if got.get("trustworthy") else
                         "轨迹未进入扩散区（MSD 仍亚扩散），D 与电导率均不可用，请加长 ps。")
        return got

    def design_formulation(self, k_target, T=298.15, n=8, pool=20000, salt=""):
        """给目标电导率推荐配方。从真实配方采样+扰动+打分，不训生成模型。"""
        assert self.cond is not None, "配方尺子未加载"
        import pandas as pd
        net, ck = self.cond
        SOLV, SALTS, RATIOS = ck["solvents"], ck["salts"], ck["ratios"]
        D = pd.read_csv(CALISOL)
        D = D[(D["k"] > 0) & D["k"].notna() & D["c"].notna()]
        if salt:
            D = D[D["salt"].astype(str).str.strip() == salt]
            assert len(D) > 50, "锂盐 %s 数据不足 %d 条" % (salt, len(D))
        rng = np.random.RandomState(0)
        seeds = D.sample(n=min(pool, len(D)), replace=True, random_state=0)
        cands, X = [], []
        for _, r in seeds.iterrows():
            sv = np.array([float(r[c]) if pd.notna(r[c]) else 0. for c in SOLV])
            if sv.sum() <= 0:
                continue
            sv = sv / sv.sum()
            nz = sv > 0
            sv[nz] = np.clip(sv[nz] * rng.uniform(0.6, 1.6, nz.sum()), 1e-3, None)
            sv = sv / sv.sum()
            c = float(r["c"]) * rng.uniform(0.7, 1.4)
            st, rt = str(r["salt"]), str(r["solvent ratio type"])
            sh = np.zeros(len(SALTS)); sh[SALTS.index(st) if st in SALTS else 0] = 1
            rh = np.zeros(len(RATIOS)); rh[RATIOS.index(rt) if rt in RATIOS else 0] = 1
            cands.append((sv, st, c, rt))
            X.append(np.concatenate([sv, sh, rh, [c, T, 1000./T]]))
        assert cands, "没有候选配方"
        X = (np.stack(X).astype(np.float32) - ck["mu"]) / ck["sd"]
        with torch.no_grad():
            p = net(torch.tensor(X)).squeeze(-1).numpy() * ck["ys"] + ck["ym"]
        pred = np.exp(p)
        seen, out = set(), []
        for i in np.argsort(np.abs(pred - k_target)):
            sv, st, c, rt = cands[i]
            top = [(SOLV[j], sv[j]) for j in np.argsort(-sv)[:4] if sv[j] > 0.02]
            key = (st, tuple(s for s, _ in top), round(c, 1))
            if key in seen:
                continue
            seen.add(key)
            out.append({"salt": st, "concentration": round(c, 2), "ratio_basis": rt,
                        "solvents": {s: round(float(v), 3) for s, v in top},
                        "predicted_k": round(float(pred[i]), 3)})
            if len(out) >= n:
                break
        return {"target_k": k_target, "T": T, "results": out,
                "scorer": {"test_r2_single_split": round(ck["test_r2"], 4),
                           "cv_r2_5fold": "全部 ≤0.29，三折为负"},
                "caveat": "★跨文献外推不可信（5折交叉验证 R² −0.03~0.29）。"
                          "仅在 CALiSol 覆盖的体系内可用，新溶剂体系不要采信"}


if __name__ == "__main__":
    cc = ClearChem(load_generator=("--nogen" not in sys.argv))
    print("\n=== 分子层：目标 gap 8.5 + lumo 0.5 ===")
    if cc.gen:
        r = cc.design_molecule({"gap": 8.5, "lumo": 0.5}, n=5)
        print("生成 %d 唯一 %d（K=%d）" % (r["n_generated"], r["n_unique"], r["k"]))
        for x in r["results"]:
            print("  %-44s SA %-5s %s" % (x["smiles"][:44], x["sa"],
                  " ".join("%s=%.2f" % (a, b) for a, b in x["predicted"].items())))
        print("  " + r["caveat"])
    print("\n=== 配方层：目标电导率 10 mS/cm @298K ===")
    f = cc.design_formulation(10.0, n=5)
    for x in f["results"]:
        print("  %-9s %.2fM  %-34s → %.2f mS/cm" % (x["salt"][:9], x["concentration"],
              " : ".join("%s %.0f%%" % (a, b*100) for a, b in x["solvents"].items())[:34],
              x["predicted_k"]))
    print("  " + f["caveat"])
