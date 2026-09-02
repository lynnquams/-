#!/usr/bin/env python3
"""ClearChem 本地命令行 —— 只用标准库，不需要 torch/GPU。

模型跑在远端（两个 27B/ether0 底座加起来 100 GB，本机装不下），
这边只发 HTTP。SSH 端口转发之后 CLEARCHEM_URL 指到 localhost 即可：

    ssh -N -L 8900:localhost:8900 <服务器>          # 另开一个终端保持
    export CLEARCHEM_URL=http://localhost:8900

用法
    clearchem_cli.py health
    clearchem_cli.py caps
    clearchem_cli.py orbitals "C1COC(=O)O1" "O=C1OC=CO1"
    clearchem_cli.py additive "O=C1OCC(F)O1"
    clearchem_cli.py predict "CCOC(=O)OCC"
    clearchem_cli.py formulate 10.0
    clearchem_cli.py design gap=8.5 n=5
    clearchem_cli.py ask "What is the formula of ethylene carbonate?"
"""
import json, os, sys, urllib.error, urllib.request

URL = os.environ.get("CLEARCHEM_URL", "http://localhost:8900").rstrip("/")
TIMEOUT = int(os.environ.get("CLEARCHEM_TIMEOUT", "600"))


def call(path, payload=None):
    url = "%s/%s" % (URL, path.lstrip("/"))
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return json.load(r)
    except urllib.error.URLError as e:
        sys.exit("连不上 %s：%s\n"
                 "  服务没起？  ssh <服务器> 'cd .../clearchem && python3 scripts/serve.py'\n"
                 "  没转发端口？ssh -N -L 8900:localhost:8900 <服务器>" % (url, e))


def show(x):
    print(json.dumps(x, ensure_ascii=False, indent=2))


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd, args = sys.argv[1], sys.argv[2:]

    if cmd == "health":
        show(call("health"))
    elif cmd in ("caps", "capabilities"):
        for k, v in call("capabilities").items():
            print("%-22s %s" % (k, v["desc"]))
            print("%-22s ⚠ %s" % ("", v["caveat"]))
    elif cmd == "orbitals":
        r = call("orbitals", {"smiles": args})
        print("%-28s %9s %9s %9s" % ("SMILES", "HOMO/eV", "LUMO/eV", "gap/eV"))
        for s, d in r.items():
            if "error" in d:
                print("%-28s %s" % (s[:28], d["error"][:40])); continue
            print("%-28s %9.3f %9.3f %9.3f" % (s[:28], d["homo"], d["lumo"], d["gap"]))
    elif cmd == "additive":
        r = call("screen_additive", {"smiles": args})
        print("参照 %s  LUMO %.3f eV" % (r["reference"], r["reference_lumo"]))
        for s, d in r["candidates"].items():
            if "lumo_below_ref_eV" not in d:
                print("  %-24s %s" % (s[:24], d.get("error", "?")[:40])); continue
            print("  %-24s 比参照低 %+.3f eV   能成膜: %s"
                  % (s[:24], d["lumo_below_ref_eV"], "是" if d["sei_forming"] else "否"))
    elif cmd == "predict":
        show(call("predict", {"smiles": args}))
    elif cmd == "formulate":
        show(call("design_formulation", {"k_target": float(args[0]),
                                         "n": int(args[1]) if len(args) > 1 else 8}))
    elif cmd == "design":
        kw = dict(a.split("=", 1) for a in args if "=" in a)
        n = int(kw.pop("n", 10))
        show(call("design_molecule", {"targets": {k: float(v) for k, v in kw.items()},
                                      "n": n}))
    elif cmd == "ask":
        r = call("ask", {"question": " ".join(args),
                         "tool": os.environ.get("TOOL", "0") == "1"})
        print(r["answer"])
    else:
        sys.exit("不认识的命令 %r，直接运行看用法" % cmd)


if __name__ == "__main__":
    main()
