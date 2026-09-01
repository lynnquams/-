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
        adp = cfg.get("adapters", {}).get("qwen", os.path.join(root, "models/clearchem-qwen"))
        assert os.path.exists(os.path.join(adp, "adapter_model.safetensors")), "适配器缺失"
        return "底座与适配器就位（推理需自行加载，见 docs/TECHNICAL.md）"
    check("知识层权重", qwen, required=False)

    print()
    if _fails:
        print("\033[1;31m自检未通过：%s\033[0m\n" % "、".join(_fails))
        sys.exit(1)
    print("\033[1;32m全部通过。\033[0m 标 — 的是可选项（缺底座时跳过，不影响其余功能）\n")


if __name__ == "__main__":
    main()
