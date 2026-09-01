"""ChemBench 官方口径重跑。

前一版有三处不符官方：
  ① 243 道 mae 数值题整类跳过（不是抽取失败，是根本没跑）
  ② 分母只算已判定题（1224/2542=0.4815），官方不剔除（1224/2785=0.4390）
  ③ 答案存 200 字符截断，37.2% 被切，无法复评

官方判定逻辑（src/chembench/metrics.py::all_correct）：
  MCQ    set(选中项) == set(target_scores 里值为 1 的项)   完全集合相等
  数值   mae(found, expected) == 0                        精确相等
  抽不出答案 → 算错，不从分母剔除
"""
import json, os, re, sys, time
from collections import Counter
from pathlib import Path
import pandas as pd, torch

MODEL = os.environ.get("MODEL", _os.path.join(_R, "bases", "qwen"))
HF = Path(_os.path.join(_R, "data", "chembench_hf"))
OUT = Path(os.environ.get("OUT", "" + _R + "/PLACEHOLDER/chembench_official_v3.json"))
# 实测答案长度均值 445、中位 587 字符：模型会先写完整推理再给结论。
# 160 token 会把它砍在推理中途，压根到不了 "ANSWER:"。
# 强制只输出答案本身：实测这样 12 题全部干净输出单字母、10/12 正确。
# 让模型"先推理再给 ANSWER"会写 1700+ 字符还不写标记，两次全量重跑都栽在这。
MAXNEW = int(os.environ.get("MAXNEW", "24"))
BATCH = int(os.environ.get("BATCH", "16"))
NVOTE = int(os.environ.get("NVOTE", "1"))    # >1 开自洽投票
TEMP = float(os.environ.get("TEMP", "0.7"))
THINK = os.environ.get("THINK", "0") == "1"   # Qwen3.5 原生思考模式
TOOL = int(os.environ.get("TOOL", "0"))     # 1 = 数值题让模型写 Python 现算
TOOL_TIMEOUT = int(os.environ.get("TOOL_TIMEOUT", "20"))
LIM = int(os.environ.get("LIM", "0"))

import glob

import os as _os
_R = _os.environ.get("CLEARCHEM_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))

frames = []
for f in sorted(glob.glob(str(HF / "*" / "train-*.parquet"))):
    d = pd.read_parquet(f)
    d["category"] = os.path.basename(os.path.dirname(f))
    frames.append(d)
allq = pd.concat(frames, ignore_index=True)
assert len(allq) > 2700, "题库不完整: %d" % len(allq)
print("题库 %d 题  %s" % (len(allq), dict(Counter(allq["preferred_score"]))), flush=True)

LETTERS = "ABCDEFGHIJKLMNOP"

def build(row):
    """返回 (prompt, gold, kind)。gold 为字母集合或数值。"""
    ex = row["examples"][0]
    q = str(ex["input"]).strip()
    ps = row["preferred_score"]
    if ps == "multiple_choice_grade":
        ts = ex["target_scores"]
        if isinstance(ts, str): ts = json.loads(ts)
        if not isinstance(ts, dict) or not ts: return None
        opts = list(ts.keys())
        gold = {LETTERS[i] for i, o in enumerate(opts) if float(ts[o]) == 1.0}
        if not gold: return None
        body = "\n".join("%s. %s" % (LETTERS[i], o) for i, o in enumerate(opts))
        p = ("Answer the following chemistry question.\n\n%s\n\n%s\n\n"
             "Reply with ONLY the letter(s) of the correct option(s). "
             "No explanation. No reasoning. Just the letter(s), "
             "comma-separated if more than one." % (q, body))
        return p, gold, "mcq"
    else:
        tgt = ex.get("target")
        if tgt is None: return None
        try: gold = float(tgt)
        except Exception: return None
        p = ("Answer the following chemistry question.\n\n%s\n\n"
             "Reply with ONLY the final numeric value. "
             "No explanation. No units. Just the number." % q)
        return p, gold, "num"

items = []
for i, row in allq.iterrows():
    b = build(row)
    if b: items.append((row["name"], row["category"], b[0], b[1], b[2]))
print("可评测 %d 题（构造失败 %d）" % (len(items), len(allq) - len(items)), flush=True)
assert len(items) > 2700, "构造后题目过少"
if LIM: items = items[:LIM]
SHARD = int(os.environ.get("SHARD", "0"))
NSHARD = int(os.environ.get("NSHARD", "1"))
if NSHARD > 1:
    assert 0 <= SHARD < NSHARD, "SHARD 越界 %d/%d" % (SHARD, NSHARD)
    items = items[SHARD::NSHARD]
    print("分片 %d/%d：本片 %d 题" % (SHARD, NSHARD, len(items)), flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map={"": 0}, trust_remote_code=True)
# ADAPTER 指向 CPT/SFT 训出的 LoRA 权重；不设就是裸底座（基线 0.5903）
ADAPTER = os.environ.get("ADAPTER", "")
if ADAPTER:
    from peft import PeftModel
    assert os.path.exists(os.path.join(ADAPTER, "adapter_model.safetensors")), \
        "adapter 不存在: " + ADAPTER
    model = PeftModel.from_pretrained(model, ADAPTER)
    print("已加载 adapter: " + ADAPTER, flush=True)
else:
    print("裸底座（无 adapter）", flush=True)
model.eval()
print("模型 %.1f GiB" % (torch.cuda.memory_allocated()/2**30), flush=True)

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def strip_think(t):
    """剥掉 <think>...</think>。没有闭合标签 = 推理被截断，返回空串（记为抽不出）。"""
    if "<think>" not in t and "</think>" not in t:
        return t
    if "</think>" in t:
        return t.split("</think>", 1)[1].strip()
    return ""


def extract(text, kind):
    text = strip_think(text)
    """输出被强制为答案本身，直接解析整段；抽不出返回 None（官方口径判错）。"""
    t = text.strip()
    m = re.search(r"ANSWER\s*[:：]\s*(.+)", t, re.I)   # 仍兼容带标记的情况
    if m:
        t = m.group(1).split("\n")[0]
    t = t.strip().strip(".").strip()
    if kind == "mcq":
        # 只在前 40 字符内找：正常输出就是 "C" 或 "A, C"
        picked = set(re.findall(r"\b([A-P])\b", t[:40].upper()))
        return picked or None
    nums = NUM_RE.findall(t[:60].replace(",", ""))
    if not nums:
        return None
    try:
        return float("%.12g" % float(nums[0]))
    except Exception:
        return None


def judge(found, gold, kind):
    if found is None: return False           # 官方：抽不出 = 错
    if kind == "mcq": return set(found) == set(gold)
    # 官方 chembench.metrics.all_correct: mae(found,expected)==0，严格相等
    return float(found) == float(gold)

def gen_texts(prompts, maxnew, think=None):
    out = []
    for i in range(0, len(prompts), BATCH):
        b = prompts[i:i + BATCH]
        msgs = [tok.apply_chat_template([{"role": "user", "content": c}], tokenize=False,
                add_generation_prompt=True, enable_thinking=THINK if think is None else think) for c in b]
        enc = tok(msgs, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(0)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=maxnew, do_sample=False,
                               pad_token_id=tok.pad_token_id)
        out += [tok.decode(g[j][enc["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip() for j in range(len(b))]
    return out

def run_snippet(code):
    """跑模型写的 Python，取最后一行数字。

    ponytail: 子进程 + 超时 + 临时目录 + 空环境，不是完整沙箱。
    代码来自本地模型回答化学题、跑在一次性容器里，风险可控；
    若要在共享或联网机器上跑，升级到容器/seccomp 隔离。
    """
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "s.py")
        open(f, "w").write(code)
        try:
            r = subprocess.run([sys.executable, f], capture_output=True, text=True,
                               timeout=TOOL_TIMEOUT, cwd=td,
                               env={"PATH": os.environ.get("PATH", ""),
                                    "HOME": td, "PYTHONHASHSEED": "0"})
        except Exception:
            return None
    if r.returncode != 0:
        return None
    m = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", r.stdout.replace(",", ""))
    return float("%.12g" % float(m[-1])) if m else None


def tool_answer(qs):
    """对数值题：先让模型写代码，跑得出数就用，跑不出退回直答。"""
    ps = [("Write a short Python program that computes the answer to this chemistry "
           "question and prints ONLY the final number.\n\n%s\n\n"
           "You may use rdkit, math, itertools, sympy. Output only code, no markdown "
           "fences, no explanation." % q) for q in qs]
    outs = gen_texts(ps, 512)
    vals = []
    for o in outs:
        code = re.sub(r"^```(?:python)?|```$", "", o.strip(), flags=re.M).strip()
        vals.append(run_snippet(code) if code else None)
    return vals


recs, t0 = [], time.time()
for i in range(0, len(items), BATCH):
    ch = items[i:i+BATCH]
    msgs = [tok.apply_chat_template([{"role": "user", "content": c[2]}],
            tokenize=False, add_generation_prompt=True, enable_thinking=THINK) for c in ch]
    enc = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=MAXNEW,
                           do_sample=NVOTE > 1, temperature=TEMP if NVOTE > 1 else None,
                           top_p=0.95 if NVOTE > 1 else None,
                           num_return_sequences=NVOTE,
                           pad_token_id=tok.pad_token_id)
    for j, c in enumerate(ch):
        outs = [tok.decode(g[j * NVOTE + k][enc["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip() for k in range(NVOTE)]
        votes = [extract(o, c[4]) for o in outs]
        # 多数表决：按抽出的答案本身投票（集合/数值都可哈希化），抽不出的不投票。
        # 官方口径要求集合完全相等，所以票必须投给整个答案而非单个选项。
        cand = [tuple(sorted(v)) if isinstance(v, set) else v for v in votes if v is not None]
        f = None
        if cand:
            win = Counter(cand).most_common(1)[0][0]
            f = set(win) if isinstance(win, tuple) else win
        ans = outs[0] if NVOTE == 1 else "|".join(outs)
        recs.append({"name": c[0], "category": c[1], "kind": c[4],
                     "answer": ans,                      # 完整存，不截断
                     "extracted": sorted(f) if isinstance(f, set) else f,
                     "gold": sorted(c[3]) if isinstance(c[3], set) else c[3],
                     "correct": bool(judge(f, c[3], c[4]))})
    if (i//BATCH) % 20 == 0:
        done = len(recs); acc = sum(r["correct"] for r in recs)/max(done,1)
        el = (time.time()-t0)/60
        print("  %4d/%d  running=%.4f  %.1fmin  eta %.0fmin"
              % (done, len(items), acc, el, el/max(done,1)*(len(items)-done)), flush=True)
        json.dump({"partial": True, "n": done, "acc": acc}, open(OUT, "w"))

if THINK:
    idx = [i for i, r in enumerate(recs) if r["extracted"] is None]
    if idx:
        print("\n强制收尾：%d 道推理超预算未出答案" % len(idx), flush=True)
        ps = []
        for i in idx:
            tail = recs[i]["answer"][-6000:]
            ps.append(items[i][2] + "\n\n[Reasoning so far]\n" + tail +
                      "\n\n[End] Based on the reasoning above, reply with ONLY the "
                      "letter(s) of the final answer, comma-separated. Nothing else.")
        outs = gen_texts(ps, 16, think=False)
        n_rec = 0
        for i, o in zip(idx, outs):
            f = extract(o, items[i][4])
            if f is not None:
                recs[i]["extracted"] = sorted(f) if isinstance(f, set) else f
                recs[i]["correct"] = bool(judge(f, items[i][3], items[i][4]))
                n_rec += 1
        print("  救回 %d 道，答对 %d 道"
              % (n_rec, sum(recs[i]["correct"] for i in idx)), flush=True)

if TOOL:
    idx = [i for i, c in enumerate(items) if c[4] == "num"]
    print("\n工具调用：%d 道数值题让模型写 Python 现算" % len(idx), flush=True)
    vals = tool_answer([items[i][2] for i in idx])
    n_fix = n_break = n_fail = 0
    for i, v in zip(idx, vals):
        if v is None:
            n_fail += 1
            continue
        was = recs[i]["correct"]
        now = judge(v, items[i][3], "num")
        if now and not was: n_fix += 1
        if was and not now: n_break += 1
        recs[i]["extracted"] = v
        recs[i]["correct"] = bool(now)
    print("  代码跑不出 %d 道（保留直答）  修好 %d  弄坏 %d  净 %+d"
          % (n_fail, n_fix, n_break, n_fix - n_break), flush=True)

n_ok = sum(r["correct"] for r in recs)
n_unext = sum(1 for r in recs if r["extracted"] is None)
by_cat, by_kind = {}, {}
for r in recs:
    by_cat.setdefault(r["category"], []).append(r["correct"])
    by_kind.setdefault(r["kind"], []).append(r["correct"])
out = {"model": MODEL, "n_total": len(recs),
       "fraction_correct_official": n_ok/len(recs), "nvote": NVOTE, "think": THINK,
       "n_unextractable": n_unext,
       "by_category": {k: sum(v)/len(v) for k, v in by_cat.items()},
       "by_kind": {k: {"n": len(v), "acc": sum(v)/len(v)} for k, v in by_kind.items()},
       "results": recs}
json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
print("\n" + "="*62)
print("官方口径 fraction_correct = %.4f  (%d/%d)" % (n_ok/len(recs), n_ok, len(recs)))
print("抽不出答案 %d 条（已按官方计为错）" % n_unext)
for k, v in out["by_kind"].items(): print("  %-5s n=%4d acc=%.4f" % (k, v["n"], v["acc"]))
print("\n分类:")
for k, v in sorted(out["by_category"].items(), key=lambda x: -x[1]):
    print("  %-24s %.4f" % (k, v))
lb = json.load(open(os.environ.get("LB", "" + _R + "/PLACEHOLDER/leaderboard.json")))
near = sorted(lb.items(), key=lambda x: -x[1]["fraction_correct"])
me = n_ok/len(recs)
print("\n官方排行榜相邻位置:")
for name, d in near:
    s = d["fraction_correct"]
    if abs(s - me) < 0.06: print("  %-30s %.4f%s" % (name, s, "  ← 我们 %.4f" % me if s < me else ""))
print("=== CB OFFICIAL DONE ===")
