"""只测 ChemBench 的 1000 道 chemical_preference 题。

为什么要单独测：全量 2785 题的最小可分辨差是 1.15pp，而 preference 只占 36%，
它涨 3pp 在总分上只体现 1.1pp —— 正好被分辨率吃掉，看不出来。
单独测这 1000 题，分辨率是 1.96*sqrt(n_discordant)/1000，灵敏得多。

这 1000 题是二选一（随机 = 0.50），训练数据是 MolSkill 去掉这 1000 对后的 4275 对。
"""
import glob, json, os, re, sys, time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import os as _os
_R = _os.environ.get("CLEARCHEM_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))


HF = os.environ.get("HF", _os.path.join(_R, "data", "chembench_hf"))
MODEL = os.environ.get("MODEL", _os.path.join(_R, "bases", "qwen"))
ADAPTER = os.environ.get("ADAPTER", "")
OUT = Path(os.environ.get("OUT", "" + _R + "/PLACEHOLDER/pref_eval.json"))
BATCH = int(os.environ.get("BATCH", "24"))
LETTERS = "ABCDEFGHIJKLMNOP"

a = pd.concat([pd.read_parquet(f) for f in
               sorted(glob.glob(HF + "/*/train-*.parquet"))], ignore_index=True)
pf = a[a["name"].astype(str).str.startswith("preference")]
assert len(pf) == 1000, "preference 题数异常 %d" % len(pf)

items = []
for _, row in pf.iterrows():
    ex = row["examples"][0]
    ts = ex["target_scores"]
    ts = json.loads(ts) if isinstance(ts, str) else ts
    opts = list(ts.keys())
    gold = {LETTERS[i] for i, o in enumerate(opts) if float(ts[o]) == 1.0}
    body = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(opts))
    p = ("Answer the following chemistry question.\n\n%s\n\n%s\n\n"
         "Reply with ONLY the letter(s) of the correct option(s). "
         "No explanation. No reasoning. Just the letter(s), "
         "comma-separated if more than one." % (str(ex["input"]).strip(), body))
    items.append((row["name"], p, gold))
assert len(items) == 1000

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map={"": 0}, trust_remote_code=True)
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
    print("adapter: %s" % ADAPTER, flush=True)
model.eval()

recs, t0 = [], time.time()
for i in range(0, len(items), BATCH):
    ch = items[i:i + BATCH]
    msgs = [tok.apply_chat_template([{"role": "user", "content": c[1]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for c in ch]
    enc = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=24, do_sample=False,
                           pad_token_id=tok.pad_token_id)
    for j, c in enumerate(ch):
        t = tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        m = re.search(r"ANSWER\s*[:：]\s*(.+)", t, re.I)
        if m:
            t = m.group(1).split("\n")[0]
        found = set(re.findall(r"\b([A-P])\b", t[:40].upper())) or None
        recs.append({"name": c[0], "gold": sorted(c[2]),
                     "extracted": sorted(found) if found else None,
                     "correct": bool(found and found == c[2])})
    if (i // BATCH) % 10 == 0:
        acc = sum(r["correct"] for r in recs) / max(len(recs), 1)
        print("  %4d/1000  running=%.4f  %.1fmin" % (len(recs), acc, (time.time()-t0)/60),
              flush=True)

acc = sum(r["correct"] for r in recs) / len(recs)
unext = sum(1 for r in recs if r["extracted"] is None)
json.dump({"adapter": ADAPTER, "n": len(recs), "acc": acc, "n_unextractable": unext,
           "results": recs}, open(OUT, "w"), ensure_ascii=False, indent=1)
print("\npreference 1000 题  正确率 %.4f  抽不出 %d  (随机=0.5000)" % (acc, unext))
print("→ %s" % OUT)
