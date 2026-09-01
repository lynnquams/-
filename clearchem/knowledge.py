"""知识层：ClearChem-Qwen 的调用接口。

ChemBench 0.6445（超 o1-preview 0.6435）是 agent 系统成绩 ——
数值题接了 Python 工具执行。纯模型（不接工具）实测 0.6316。
这里把两种模式都提供，默认开工具。

用法：
    from clearchem.knowledge import ChemQwen
    q = ChemQwen()
    q.ask("What is the molecular formula of ethylene carbonate?")
    q.ask("How many distinct 1H NMR signals does toluene have?", tool=True)
"""
import json, os, re, subprocess, sys, tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_CFG = {}
_cfg_file = os.path.join(_HERE, "config.json")
if os.path.exists(_cfg_file):
    _CFG = json.load(open(_cfg_file))
ROOT = _CFG.get("root") or os.environ.get("CLEARCHEM_ROOT") or os.path.dirname(_HERE)
BASE = _CFG.get("bases", {}).get("qwen") or os.path.join(ROOT, "bases", "qwen")
ADAPTER = _CFG.get("adapters", {}).get("qwen") or os.path.join(ROOT, "models", "clearchem-qwen")


class ChemQwen:
    def __init__(self, base=None, adapter=None, device_map="auto", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        base = base or BASE
        adapter = adapter or ADAPTER
        assert os.path.exists(os.path.join(base, "config.json")), (
            "底座不在 %s。用 WITH_QWEN=1 bash scripts/deploy.sh 自动下载，"
            "或手动放置后传 base= 参数" % base)
        assert os.path.exists(os.path.join(adapter, "adapter_model.safetensors")), (
            "适配器不在 %s。先跑 bash scripts/assemble_weights.sh 合并分卷" % adapter)
        self.tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        m = AutoModelForCausalLM.from_pretrained(
            base, torch_dtype=getattr(torch, dtype), device_map=device_map,
            trust_remote_code=True)
        self.model = PeftModel.from_pretrained(m, adapter)
        self.model.eval()
        self.dev = next(self.model.parameters()).device
        print("ChemQwen 就绪  底座 %s  适配器 %s" % (os.path.basename(base),
                                                    os.path.basename(adapter)), flush=True)

    # ---- 基础生成 ----
    def _gen(self, prompts, max_new_tokens=64, think=False):
        import torch
        msgs = [self.tok.apply_chat_template([{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=think) for p in prompts]
        enc = self.tok(msgs, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.dev)
        with torch.no_grad():
            g = self.model.generate(**enc, max_new_tokens=max_new_tokens,
                                    do_sample=False, pad_token_id=self.tok.pad_token_id)
        return [self.tok.decode(g[i][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()
                for i in range(len(prompts))]

    # ---- Python 工具执行 ----
    @staticmethod
    def _run_code(code, timeout=20):
        """跑模型写的代码，取最后一个数字。

        ponytail: 子进程 + 超时 + 临时目录 + 最小环境，不是完整沙箱。
        代码来自本地模型回答化学题；若要在共享或联网机器上跑，升级到容器/seccomp。
        """
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "s.py")
            open(f, "w").write(code)
            try:
                r = subprocess.run([sys.executable, f], capture_output=True, text=True,
                                   timeout=timeout, cwd=td,
                                   env={"PATH": os.environ.get("PATH", ""),
                                        "HOME": td, "PYTHONHASHSEED": "0"})
            except Exception:
                return None
        if r.returncode != 0:
            return None
        m = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", r.stdout.replace(",", ""))
        if not m:
            return None
        # 规整到 12 位有效数字：官方判定是严格浮点相等，
        # -0.6000000000000005 这类尾数会被判错
        return float("%.12g" % float(m[-1]))

    def ask(self, question, tool=False, max_new_tokens=64):
        """问一个化学问题。

        tool=True 时先让模型写 Python 求解并执行，跑不出再退回直答。
        计算类问题（数几种信号、算分子量、枚举异构体）建议开工具：
        实测 ChemBench 数值题 0.4362 → 0.6214。
        """
        if not tool:
            return self._gen([question], max_new_tokens)[0]
        code_prompt = ("Write a short Python program that computes the answer to this "
                       "chemistry question and prints ONLY the final number.\n\n%s\n\n"
                       "You may use rdkit, math, itertools, sympy. Output only code, "
                       "no markdown fences, no explanation." % question)
        raw = self._gen([code_prompt], 512)[0]
        code = re.sub(r"^```(?:python)?|```$", "", raw.strip(), flags=re.M).strip()
        val = self._run_code(code) if code else None
        if val is not None:
            return {"answer": val, "via": "python", "code": code}
        return {"answer": self._gen([question], max_new_tokens)[0],
                "via": "direct", "code": code, "note": "代码未跑出结果，已退回直答"}

    def choose(self, question, options):
        """多选题。返回选中的字母集合。

        注意：模型在多答案题上系统性过度多选（实测 394 道多答案题里 45.2% 选多了）。
        四种推理侧裁剪法实测全都不如不裁，见 docs/FAILURES.md。
        """
        L = "ABCDEFGHIJKLMNOP"
        body = "\n".join("%s. %s" % (L[i], o) for i, o in enumerate(options))
        p = ("Answer the following chemistry question.\n\n%s\n\n%s\n\n"
             "Reply with ONLY the letter(s) of the correct option(s). "
             "No explanation. No reasoning. Just the letter(s), "
             "comma-separated if more than one." % (question, body))
        t = self._gen([p], 24)[0]
        m = re.search(r"ANSWER\s*[:：]\s*(.+)", t, re.I)
        if m:
            t = m.group(1).split("\n")[0]
        return sorted(set(re.findall(r"\b([A-P])\b", t[:40].upper())))


if __name__ == "__main__":
    q = ChemQwen()
    print("\n【直答】")
    print(" ", q.ask("What is the molecular formula of ethylene carbonate?"))
    print("\n【工具】")
    r = q.ask("How many distinct 1H NMR signals does toluene have?", tool=True)
    print("  答案 %s（via %s）" % (r["answer"], r["via"]))
    print("\n【多选】")
    print(" ", q.choose("Which of these is a cyclic carbonate?",
                        ["dimethyl carbonate", "ethylene carbonate",
                         "ethyl acetate", "tetrahydrofuran"]))
