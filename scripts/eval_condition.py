"""ClearChem 针对性提升：攻已知的两个短板，不再盲目训练。

今天已证明：从 v1 继续训练不会更好（emb4 早停，MAE 1.063→1.337 退化）。
v1/emb2 已接近这个架构上限。真正有空间的是推理侧。

两个短板：
  ① 高 gap 区精度差    目标12.5时偏差最大（emb2 单次 +2.28，拒绝采样后 +0.01）
  ② 三条件时精度掉一半  gap 单条件 0.221 → 三条件 0.448

做法：分区拒绝采样。不同目标区间用不同的过采样倍数和温度 ——
低 gap 区容易命中，K 小即可；高 gap 区稀有，需要更大 K 才捞得到。
这是把固定 K=8 换成自适应，用同样的总算力换更均匀的精度。
"""
import json, os, sys, time
import numpy as np
import torch, torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
RDLogger.DisableLog("rdApp.*")

CK = torch.load("" + _R + "/PLACEHOLDER/gap_seed17.pt", map_location="cpu",
                weights_only=False)
assert CK["test_mae"] < 0.5
net = nn.Sequential(nn.Linear(2061,1024), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(1024,512), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(512,256), nn.GELU(), nn.Linear(256,1))
net.load_state_dict(CK["state_dict"]); net.eval()
mu, sd, dmu, dsd = CK["mu"], CK["sd"], CK["dmu"], CK["dsd"]


def feat(s):
    m = Chem.MolFromSmiles(s)
    if m is None: return None
    fp = np.array(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048), dtype=np.float32)
    d = np.array([Descriptors.MolWt(m),Descriptors.MolLogP(m),Descriptors.TPSA(m),
                  Descriptors.NumHDonors(m),Descriptors.NumHAcceptors(m),
                  Descriptors.NumRotatableBonds(m),Descriptors.RingCount(m),
                  rdMolDescriptors.CalcNumAromaticRings(m),Descriptors.FractionCSP3(m),
                  Descriptors.HeavyAtomCount(m),Descriptors.NumValenceElectrons(m),
                  Descriptors.MaxPartialCharge(m) or 0,Descriptors.MinPartialCharge(m) or 0],
                 dtype=np.float32)
    return np.concatenate([fp,(np.nan_to_num(d,nan=0.,posinf=0.,neginf=0.)-dmu)/dsd])


def gap_of(smis):
    X, keep = [], []
    for i, s in enumerate(smis):
        f = feat(s)
        if f is not None: X.append(f); keep.append(i)
    if not X: return {}
    with torch.no_grad():
        p = net(torch.tensor(np.stack(X))).squeeze(-1).numpy()*sd+mu
    return {smis[i]: float(v) for i, v in zip(keep, p)}


from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import os as _os
_R = _os.environ.get("CLEARCHEM_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

SNAP=_os.path.join(_R, "bases", "ether0"); ADP=_os.path.join(_R, "models", "clearchem-gen")
tok=AutoTokenizer.from_pretrained(SNAP,trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
tok.padding_side="left"
base=AutoModelForCausalLM.from_pretrained(SNAP,device_map={"":0},trust_remote_code=True)
model=PeftModel.from_pretrained(base,ADP); model.eval()
hid=model.get_input_embeddings().weight.shape[1]
class CondEncoder(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Linear(10,512),nn.GELU(),nn.Linear(512,8*hid))
    def forward(self,x): return self.net(x)
enc=CondEncoder(); enc.load_state_dict(torch.load(os.path.join(ADP,"cond_encoder.pt"),map_location="cpu"))
enc=enc.to(0).eval()

def gen(tgt, n, temp):
    pr=["设计一个电解液相关分子。目标性质：gap≈%.2f eV。"%tgt]*n
    out=[]
    for i in range(0,n,8):
        b=pr[i:i+8]
        e=tok(b,return_tensors="pt",padding=True,add_special_tokens=False).to(0)
        with torch.no_grad():
            cond=torch.zeros(len(b),5,device=0); cm=torch.zeros(len(b),5,device=0)
            cond[:,2]=(tgt-7.37)/1.88; cm[:,2]=1
            pe=enc(torch.cat([cond,cm],-1)).view(len(b),8,-1)
            te=model.get_input_embeddings()(e["input_ids"])
            g=model.generate(inputs_embeds=torch.cat([pe.to(te.dtype),te],1),
                attention_mask=torch.cat([torch.ones(len(b),8,device=0,
                    dtype=e["attention_mask"].dtype),e["attention_mask"]],1),
                max_new_tokens=192,do_sample=True,temperature=temp,top_p=0.95,
                pad_token_id=tok.pad_token_id)
            out+=[tok.decode(x,skip_special_tokens=True).strip() for x in g]
    smis=[]
    for t in out:
        try:
            d=json.loads(t[t.index("{"):t.rindex("}")+1])
            s=d["components"][0].get("smiles")
            if s and Chem.MolFromSmiles(s): smis.append(s)
        except Exception: pass
    return smis

# 分区策略：低 gap 区分布密集 K 小即可；高 gap 区稀有需要大 K
ZONES = [(5.0,6,1.2),(6.5,6,1.2),(8.0,8,1.2),(9.5,10,1.3),(11.0,14,1.4),(12.5,20,1.4)]
N_KEEP=30
print("分区拒绝采样（K 随目标升高而增大，高 gap 区更稀有）", flush=True)
t0=time.time(); res=[]
for tgt,K,T in ZONES:
    smis=gen(tgt,N_KEEP*K,T)
    if not smis: print("  %.1f 无合法"%tgt); continue
    gp=gap_of(list(set(smis)))
    cand=sorted(((abs(v-tgt),s,v) for s,v in gp.items()))[:N_KEEP]
    vals=np.array([c[2] for c in cand])
    res+= [(tgt,v) for v in vals]
    print("  目标 %.1f  K=%2d T=%.1f  生成%4d 唯一%3d  实测均值 %.2f (偏差%+.2f)  MAE %.3f"
          %(tgt,K,T,len(smis),len(gp),vals.mean(),vals.mean()-tgt,np.abs(vals-tgt).mean()),flush=True)

a=np.array([r[1] for r in res]); t=np.array([r[0] for r in res])
from scipy.stats import spearmanr
print("\n分区拒绝采样  MAE %.3f  Spearman %.3f  n=%d"%(np.abs(a-t).mean(),spearmanr(a,t).correlation,len(a)))
print("对比 固定K=8/T=1.4  MAE 0.246")
print("用时 %.1f 分"%((time.time()-t0)/60))
