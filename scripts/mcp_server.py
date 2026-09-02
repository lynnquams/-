#!/usr/bin/env python3
"""ClearChem MCP 服务器 —— 让本地 agent 直接调用整套系统。

跑在本机，只用 Python 标准库（不需要 torch/GPU/pip 装任何东西）；
真正的计算转发到远端 ClearChem HTTP 服务，那里有两个模型和五个引擎。

    本地 agent  ──stdio/JSON-RPC──▶  本文件  ──HTTP──▶  远端 ClearChem
                                                        ChemQwen 27B
                                                        ClearChem-Gen
                                                        xTB / MACE / ABACUS

接入方式（Claude Code / 任何支持 MCP 的 agent）：
    在 agent 的 MCP 配置里加一条
    {"clearchem": {"command": "python3",
                   "args": ["<绝对路径>/scripts/mcp_server.py"],
                   "env": {"CLEARCHEM_URL": "http://localhost:8900"}}}

远端服务需先起，并做端口转发：
    ssh <服务器> 'cd .../clearchem && python3 scripts/serve.py'
    ssh -N -L 8900:localhost:8900 <服务器>

每个工具的描述里都写明了实测边界 —— agent 看得到什么可信、什么不可信，
不会把"电解液分子用性质尺子"这种已知会错的用法当成可行。
"""
import json, os, sys, urllib.error, urllib.request

URL = os.environ.get("CLEARCHEM_URL", "http://localhost:8900").rstrip("/")
TIMEOUT = int(os.environ.get("CLEARCHEM_TIMEOUT", "900"))


def http(path, payload=None):
    req = urllib.request.Request(
        "%s/%s" % (URL, path.lstrip("/")),
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


TOOLS = [
    {"name": "orbitals",
     "description": ("电解液分子的 HOMO/LUMO/带隙（GFN2-xTB，约 1 秒/分子）。"
                     "电解液分子一律用这个，不要用 predict_properties —— "
                     "后者在碳酸酯上电化学排序只有 2/6（随机是 3/6），"
                     "会把 VC/FEC 判成比 EC 更难还原，与成膜机理相反。"
                     "本工具同一套检验 6/6。气相单构象，只作同类分子相对比较。"),
     "inputSchema": {"type": "object", "required": ["smiles"], "properties": {
         "smiles": {"type": "array", "items": {"type": "string"},
                    "description": "SMILES 列表"}}}},

    {"name": "screen_additive",
     "description": ("判断分子能否做 SEI 成膜添加剂：算它的 LUMO 比参照溶剂低多少。"
                     "低得越多越容易先还原成膜。实测参照点：VC 比 EC 低 0.372 eV、"
                     "FEC 低 0.566 eV（文献 0.3~0.9，吻合）。"),
     "inputSchema": {"type": "object", "required": ["smiles"], "properties": {
         "smiles": {"type": "array", "items": {"type": "string"}},
         "reference": {"type": "string", "default": "C1COC(=O)O1",
                       "description": "参照溶剂 SMILES，默认 EC"}}}},

    {"name": "predict_properties",
     "description": ("五把性质尺子（gap/homo/lumo/ip/ea），毫秒级，适合通用小分子粗筛。"
                     "⚠ 电解液分子上不可用：碳酸酯 LUMO 真实跨度 1.17 eV 被压成 0.27 eV，"
                     "而尺子自身 MAE 就是 0.28 eV —— 要分辨的差距比工具误差还小。"
                     "碳酸酯/醚/砜类请改用 orbitals。"),
     "inputSchema": {"type": "object", "required": ["smiles"], "properties": {
         "smiles": {"type": "array", "items": {"type": "string"}}}}},

    {"name": "design_molecule",
     "description": ("按目标性质生成候选分子（ClearChem-Gen，ether0 底座）。"
                     "实测条件遵循 MAE 0.109 eV，六个目标点偏差全在 ±0.03；"
                     "Novelty 0.928、Validity 0.992、SA 3.40。"),
     "inputSchema": {"type": "object", "required": ["targets"], "properties": {
         "targets": {"type": "object", "description": "如 {\"gap\": 8.5}"},
         "n": {"type": "integer", "default": 10},
         "temp": {"type": "number", "default": 1.4}}}},

    {"name": "design_formulation",
     "description": ("按目标电导率推荐配方（盐 + 溶剂 + 浓度）。"
                     "⚠ 只在 CALiSol 覆盖的 14 种锂盐 × 38 种溶剂内可信："
                     "按文献切分的 5 折交叉验证 R² 全部 ≤0.29（三折为负），"
                     "换新溶剂体系给的是瞎猜。"),
     "inputSchema": {"type": "object", "required": ["k_target"], "properties": {
         "k_target": {"type": "number", "description": "目标电导率 mS/cm"},
         "T": {"type": "number", "default": 298.15},
         "n": {"type": "integer", "default": 8},
         "salt": {"type": "string", "default": ""}}}},

    {"name": "simulate_conductivity",
     "description": ("用 MACE 分子动力学算电解液电导率，不依赖实验数据，"
                     "因而能算尺子外推不了的新体系。约 3.4 小时/配方。"
                     "⚠ 不能用于配方排序：同体系换随机种子实测 D 相差 1.84~2.20 倍，"
                     "而 EC:DMC 与 PC 的真实差距只有 1.72 倍 —— 自身波动大于要分辨的差距，"
                     "实测排序会翻转。单次结果只能读数量级。"),
     "inputSchema": {"type": "object", "required": ["comp"], "properties": {
         "comp": {"type": "object", "description": "溶剂组成，如 {\"EC\":10,\"DMC\":10}"},
         "rho": {"type": "number", "default": 1.20, "description": "目标密度 g/cm³"},
         "n_ion": {"type": "integer", "default": 6},
         "ps": {"type": "integer", "default": 1000, "description": "轨迹长度 ps"}}}},

    {"name": "ask_chemistry",
     "description": ("化学知识问答（ChemQwen，Qwen3.8-27B + LoRA）。"
                     "ChemBench 官方口径 0.6445（接 Python 工具）/ 0.6316（纯模型），"
                     "2,785 题全量。tool=true 时数值题会让模型写 Python 现算，"
                     "实测数值题正确率从 0.46 提到 0.60。"),
     "inputSchema": {"type": "object", "required": ["question"], "properties": {
         "question": {"type": "string"},
         "tool": {"type": "boolean", "default": False},
         "max_new_tokens": {"type": "integer", "default": 64}}}},

    {"name": "clearchem_health",
     "description": "查服务状态：哪些模型已载入、运行多久、是不是轻量档。",
     "inputSchema": {"type": "object", "properties": {}}},
]

ROUTE = {"orbitals": "orbitals", "screen_additive": "screen_additive",
         "predict_properties": "predict", "design_molecule": "design_molecule",
         "design_formulation": "design_formulation",
         "simulate_conductivity": "simulate_conductivity", "ask_chemistry": "ask"}


def handle(req):
    m, rid = req.get("method"), req.get("id")
    if m == "initialize":
        return {"protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "clearchem", "version": "1.0"}}
    if m == "tools/list":
        return {"tools": TOOLS}
    if m == "tools/call":
        p = req.get("params", {})
        name, args = p.get("name"), p.get("arguments") or {}
        try:
            if name == "clearchem_health":
                out = http("health")
            elif name in ROUTE:
                out = http(ROUTE[name], args)
            else:
                return {"isError": True, "content": [
                    {"type": "text", "text": "没有这个工具：%s" % name}]}
        except urllib.error.URLError as e:
            return {"isError": True, "content": [{"type": "text", "text":
                    "连不上 ClearChem 服务 %s：%s\n"
                    "服务没起或端口没转发：ssh -N -L 8900:localhost:8900 <服务器>"
                    % (URL, e)}]}
        except Exception as e:                       # 远端 500 也要如实回给 agent
            return {"isError": True, "content": [
                {"type": "text", "text": "调用失败：%s" % str(e)[:400]}]}
        return {"content": [{"type": "text",
                             "text": json.dumps(out, ensure_ascii=False, indent=1)}]}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        res = handle(req)
        if req.get("id") is None:        # 通知类消息不回
            continue
        msg = {"jsonrpc": "2.0", "id": req["id"]}
        if res is None:
            msg["error"] = {"code": -32601, "message": "未实现的方法 %s" % req.get("method")}
        else:
            msg["result"] = res
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
