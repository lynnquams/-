"""ClearChem HTTP 服务 —— 让整套系统能从别的机器调用。

ClearChem 是两个模型加多个计算引擎，不是单个大模型：
    ChemQwen        27B + LoRA      知识问答
    ClearChem-Gen   ether0 + LoRA   条件分子生成
    五把尺子 / 电导率尺子 / xTB / MACE / ABACUS 执行器

模型加载一次常驻，避免每次调用重新载入 100 GB 权重。
启动：python3 scripts/serve.py            （默认 0.0.0.0:8900）
      LIGHT=1 python3 scripts/serve.py    （只起轻量档，不载两个大模型）
"""
import json, os, sys, time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

LIGHT = os.environ.get("LIGHT", "0") == "1"
PORT = int(os.environ.get("PORT", "8900"))

app = FastAPI(title="ClearChem", version="1.0")
_cc = None
_qwen = None
_t0 = time.time()


def cc():
    global _cc
    if _cc is None:
        from clearchem import ClearChem
        _cc = ClearChem(load_generator=not LIGHT)
    return _cc


def qwen():
    """知识层单独懒加载 —— 27B 载入要几分钟，没人问就不载。"""
    global _qwen
    if _qwen is None:
        if LIGHT:
            raise HTTPException(503, "轻量档未载入知识层，去掉 LIGHT=1 重启")
        from clearchem.knowledge import ChemQwen
        _qwen = ChemQwen()
    return _qwen


class Smiles(BaseModel):
    smiles: List[str]

class Design(BaseModel):
    targets: Dict[str, float]
    n: int = 10
    temp: float = 1.4

class Formulation(BaseModel):
    k_target: float
    T: float = 298.15
    n: int = 8
    salt: str = ""

class Additive(BaseModel):
    smiles: List[str]
    reference: str = "C1COC(=O)O1"

class Ask(BaseModel):
    question: str
    tool: bool = False
    max_new_tokens: int = 64

class MD(BaseModel):
    comp: Dict[str, int]
    rho: float = 1.20
    n_ion: int = 6
    ps: int = 1000


@app.get("/health")
def health():
    return {"ok": True, "light": LIGHT, "uptime_s": round(time.time() - _t0, 1),
            "loaded": {"clearchem": _cc is not None, "qwen": _qwen is not None}}


@app.get("/capabilities")
def capabilities():
    """每个能力都带实测边界 —— 调用方要能看到什么可信什么不可信。"""
    return {
        "predict": {"desc": "五把性质尺子", "caveat":
                    "电解液分子上失效（电化学排序 2/6），碳酸酯请走 /orbitals"},
        "orbitals": {"desc": "GFN2-xTB 轨道能级", "caveat":
                     "电化学排序 6/6，气相单构象，只作同类分子相对比较"},
        "screen_additive": {"desc": "成膜添加剂筛选",
                            "caveat": "实测 VC 比 EC 低 0.372 eV、FEC 低 0.566 eV"},
        "design_molecule": {"desc": "条件分子生成",
                            "caveat": "条件遵循 MAE 0.109 eV（六点偏差 ±0.03）"},
        "design_formulation": {"desc": "配方推荐", "caveat":
                               "跨文献 5 折 R² 全部 ≤0.29，只在 CALiSol 覆盖范围内可信"},
        "simulate_conductivity": {"desc": "MACE 分子动力学", "caveat":
                                  "换种子 D 相差 1.84~2.20 倍，单次运行不能用于配方排序"},
        "ask": {"desc": "ChemQwen 化学问答",
                "caveat": "ChemBench 0.6445（接 Python 工具）/ 0.6316（纯模型）"},
    }


@app.post("/predict")
def predict(r: Smiles):
    return cc().predict(r.smiles)

@app.post("/orbitals")
def orbitals(r: Smiles):
    return cc().orbitals(r.smiles)

@app.post("/screen_additive")
def screen_additive(r: Additive):
    return cc().screen_additive(r.smiles, r.reference)

@app.post("/design_molecule")
def design_molecule(r: Design):
    return cc().design_molecule(r.targets, n=r.n, temp=r.temp)

@app.post("/design_formulation")
def design_formulation(r: Formulation):
    return cc().design_formulation(k_target=r.k_target, T=r.T, n=r.n, salt=r.salt)

@app.post("/simulate_conductivity")
def simulate_conductivity(r: MD):
    return cc().simulate_conductivity(r.comp, rho=r.rho, n_ion=r.n_ion, ps=r.ps)

@app.post("/ask")
def ask(r: Ask):
    return {"answer": qwen().ask(r.question, tool=r.tool,
                                 max_new_tokens=r.max_new_tokens)}


if __name__ == "__main__":
    print("ClearChem 服务启动 · 端口 %d · %s" % (PORT, "轻量档" if LIGHT else "完整档"))
    print("  两个模型：ChemQwen(27B) + ClearChem-Gen(ether0)，均为懒加载")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
