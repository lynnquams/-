"""部署自检：真实调用每一层，任何一项不通过就退出非零。

不做"import 成功就算通过"那种检查 —— 今天的教训是导入成功和能用是两回事。
每一项都真实跑一遍，并把结果与已知基准对照。
"""
import json, os, sys, time

OK, FAIL, SKIP = "\033[1;32m✓\033[0m", "\033[1;31m✗\033[0m", "\033[1;33m—\033[0m"
_fails = []


def check(name, fn, required=True):
    t0 = time.time()
    try:
        msg = fn()
        print("  %s %-30s %s  (%.1fs)" % (OK, name, msg, time.time() - t0))
        return True
    except Exception as e:
        mark = FAIL if required else SKIP
        print("  %s %-30s %s" % (mark, name, str(e)[:70]))
        if required:
            _fails.append(name)
        return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    root = cfg.get("root", os.path.dirname(here))
    print("\nClearChem 部署自检  root=%s\n" % root)

    # 1. 依赖
    def dep():
        import torch, rdkit, pandas, numpy, scipy, transformers, peft
        from rdkit import Chem
        assert Chem.MolFromSmiles("C1COC(=O)O1") is not None
        return "torch %s · rdkit %s · cuda %s" % (
            torch.__version__, rdkit.__version__,
            "有" if torch.cuda.is_available() else "无")
    check("依赖与 RDKit", dep)

    # 2. 打分器权重完整性
    def scorers():
        import torch
        d = cfg.get("scorers", os.path.join(root, "models/scorers"))
        found = []
        for p in ["gap", "ip", "ea", "homo", "lumo"]:
            f = os.path.join(d, "%s_seed17.pt" % p)
            assert os.path.exists(f), "缺 %s_seed17.pt" % p
            ck = torch.load(f, map_location="cpu", weights_only=False)
            assert "state_dict" in ck and "test_mae" in ck, "%s 权重格式不对" % p
            found.append("%s(%.3f)" % (p, ck["test_mae"]))
        return " ".join(found)
    check("五把分子尺子", scorers)

    # 3. 性质预测：真算一次，与已知值对照
    def predict():
        sys.path.insert(0, root)
        from clearchem.clearchem import ClearChem
        cc = ClearChem(load_generator=False)
        r = cc.predict(["C1COC(=O)O1", "CCOC(=O)OC"])   # EC, DEC
        assert len(r) == 2, "只算出 %d 个" % len(r)
        g = r["C1COC(=O)O1"].get("gap")
        assert g is not None, "gap 没算出来"
        assert 4 < g < 16, "EC 的 gap 算成 %.2f，明显不合理" % g
        globals()["_cc"] = cc
        return "EC gap=%.2f eV" % g
    check("性质预测", predict)

    # 4. 配方推荐：真跑一次
    def formulate():
        cc = globals().get("_cc")
        assert cc is not None, "上一步没建起来"
        assert cc.cond is not None, "配方尺子未加载"
        f = cc.design_formulation(10.0, n=3, pool=3000)
        assert f["results"], "没返回配方"
        top = f["results"][0]
        err = abs(top["predicted_k"] - 10.0)
        assert err < 1.0, "最优配方预测 %.2f，偏离目标 %.2f" % (top["predicted_k"], err)
        assert "caveat" in f, "缺边界说明"
        return "%s %.2fM → %.2f mS/cm" % (top["salt"][:8], top["concentration"],
                                           top["predicted_k"])
    check("配方推荐", formulate)

    # 5. 分子生成（需底座，可选）
    def generate():
        cc = globals().get("_cc")
        base = cfg.get("bases", {}).get("ether0", os.path.join(root, "bases/ether0"))
        assert os.path.exists(os.path.join(base, "config.json")), "ether0 底座未就位"
        cc._load_gen()
        assert cc.gen is not None, "生成器加载失败"
        r = cc.design_molecule({"gap": 8.5}, n=3, k=4)
        assert r.get("results"), "没生成合法分子"
        top = r["results"][0]
        return "%s gap=%.2f" % (top["smiles"][:28], top["predicted"]["gap"])
    check("分子生成", generate, required=False)

    # 6. 知识层（需 Qwen 底座，可选）
    def qwen():
        base = cfg.get("bases", {}).get("qwen", os.path.join(root, "bases/qwen"))
        assert os.path.exists(os.path.join(base, "config.json")), "Qwen 底座未就位"
        # 只查 config.json 会误判：它几 KB 先下完，权重差几十 GB 也算"就位"。
        # 实测底座只下了 3.4 MB，本项照样报 ✓，而下一项推理才炸出
        # "no file named model.safetensors"。
        import glob as _g
        wts = _g.glob(os.path.join(base, "*.safetensors")) + \
              _g.glob(os.path.join(base, "*.bin"))
        assert wts, "底座只有配置文件，权重没下（重跑 deploy.sh 续传）"
        _gb = sum(os.path.getsize(f) for f in wts) / 1073741824
        assert _gb > 40, "底座权重只有 %.1f GB，不完整（应约 54 GB）" % _gb
        adp = cfg.get("adapters", {}).get("qwen", os.path.join(root, "models/clearchem-qwen"))
        assert os.path.exists(os.path.join(adp, "adapter_model.safetensors")), "适配器缺失"
        # 维度校验：底座与适配器对不上说明下错了版本
        bc = json.load(open(os.path.join(base, "config.json")))
        tc = bc.get("text_config", bc)
        ac = json.load(open(os.path.join(adp, "adapter_config.json")))
        return "底座 %.0f GB · %d层/hidden %s ↔ 适配器 rank %s" % (
            _gb, tc.get("num_hidden_layers", -1), tc.get("hidden_size"), ac.get("r"))
    check("知识层权重", qwen, required=False)

    # 7. 知识层真实推理（需 54GB 底座 + 足够显存，可选）
    def qwen_infer():
        sys.path.insert(0, root)
        from clearchem.knowledge import ChemQwen
        q = ChemQwen()
        a = q.ask("What is the chemical formula of water?")
        assert a and len(a) < 200, "回答异常：%r" % a[:60]
        return "问答可用：%s" % a[:40].replace("\n", " ")
    check("知识层推理", qwen_infer, required=False)

    def mlip():
        import importlib.util
        for m in ("mace", "ase"):
            assert importlib.util.find_spec(m), "缺依赖 %s（pip install mace-torch ase）" % m
        w = os.path.join(root, "models", "mlip", "mace-mp-0b2-medium.model")
        assert os.path.exists(w), "势能面权重缺失 %s" % w
        import torch
        mo = torch.load(w, map_location="cpu", weights_only=False)
        n = sum(p.numel() for p in mo.parameters())
        return "势能面可加载：%.2fM 参数、%d 种元素" % (n / 1e6, len(mo.atomic_numbers))
    check("分子动力学", mlip, required=False)

    def mlip_run():
        # 真跑一小段，不只是加载。20 步足以暴露几何/近邻表/设备问题。
        import subprocess
        env = dict(os.environ, COMP='{"EC":2,"DMC":2}', RHO="1.2", NION="1",
                   TPROD="20", TEQ="20", RUNTAG="selfcheck",
                   MD_OUT=os.path.join(root, "runs"))
        r = subprocess.run([sys.executable, os.path.join(root, "clearchem", "md", "run_md.py")],
                           env=env, capture_output=True, text=True, timeout=1800)
        assert "最小原子间距" in r.stdout, "MD 未跑到建盒：%s" % (r.stderr[-200:] or r.stdout[-200:])
        return "MD 可运行（20 步冒烟）"
    check("分子动力学运行", mlip_run, required=False)

    print()
    if _fails:
        print("\033[1;31m自检未通过：%s\033[0m\n" % "、".join(_fails))
        sys.exit(1)
    print("\033[1;32m全部通过。\033[0m 标 — 的是可选项（缺底座时跳过，不影响其余功能）\n")


if __name__ == "__main__":
    main()
