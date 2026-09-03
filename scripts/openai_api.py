"""ChemQwen 的 OpenAI 兼容接口 —— 让 pi 之类的 agent 直接把它当对话模型。

为什么值得做：ChemQwen 是完整的 Qwen3.5-27B 加化学 LoRA，
工具调用能力实测完好（裸底座与 v1 适配器都能正确发 tool_call，
简单问题也不会乱调）。所以整条链可以全用自己的东西：

    pi ──OpenAI 协议──▶ 本接口 ──▶ ChemQwen（对话+工具决策）
                                  └─▶ ClearChem 九个工具（xTB/生成器/MACE/尺子）

启动：python3 scripts/openai_api.py          默认 0.0.0.0:8901
pi 侧：pi --provider openai --model qwen-chem \
          --api-key any  （base url 指到本接口）
"""
import json, os, re, sys, time, uuid
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

BASE = os.environ.get("QWEN_BASE", "")
ADAPTER = os.environ.get("QWEN_ADAPTER", "")
PORT = int(os.environ.get("PORT", "8901"))
MODEL_ID = os.environ.get("MODEL_ID", "qwen-chem")

app = FastAPI(title="ChemQwen OpenAI-compatible")
_tok = _model = None


def _paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = {}
    p = os.path.join(root, "clearchem", "config.json")
    if os.path.exists(p):
        cfg = json.load(open(p))
    base = BASE or cfg.get("bases", {}).get("qwen") or os.path.join(root, "bases/qwen")
    adp = ADAPTER or cfg.get("adapters", {}).get("qwen") or \
        os.path.join(root, "models/clearchem-qwen")
    return base, adp


def load():
    global _tok, _model
    if _model is not None:
        return _tok, _model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base, adp = _paths()
    print("载入 %s + %s" % (base, adp), flush=True)
    _tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    m = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16,
                                             device_map={"": 0}, trust_remote_code=True)
    if os.path.exists(os.path.join(adp, "adapter_model.safetensors")):
        m = PeftModel.from_pretrained(m, adp)
    _model = m.eval()
    print("就绪", flush=True)
    return _tok, _model


class Msg(BaseModel):
    role: str
    content: Optional[Any] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: List[Msg]
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    stream: bool = False


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [
        {"id": MODEL_ID, "object": "model", "owned_by": "clearchem"}]}


def _parse_tool_calls(text):
    """从输出里抠出工具调用。

    Qwen3.5 用的是 XML 参数格式，不是 JSON：
        <tool_call>
        <function=clearchem_orbitals>
        <parameter=smiles>
        ["C1COC(=O)O1"]
        </parameter>
        </function>
        </tool_call>
    第一版只认 {"name":..., "arguments":...}，结果模型明明调了工具却解析不出来，
    finish_reason 报 stop、tool_calls 是 None，agent 完全看不到。
    两种格式都要认。
    """
    calls = []
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        body = m.group(1).strip()

        # 格式一：直接是 JSON
        try:
            d = json.loads(body)
            calls.append(_mk(d.get("name"), d.get("arguments", {})))
            continue
        except json.JSONDecodeError:
            pass

        # 格式二：XML —— <function=名字> 里套若干 <parameter=键>值</parameter>
        fm = re.search(r"<function=([\w.\-]+)>(.*?)</function>", body, re.S)
        if not fm:
            fm = re.search(r"<function=([\w.\-]+)>(.*)", body, re.S)   # 结束标签可能缺
        if not fm:
            continue
        name, inner = fm.group(1), fm.group(2)
        args = {}
        for pm in re.finditer(r"<parameter=([\w.\-]+)>\s*(.*?)\s*(?:</parameter>|$)",
                              inner, re.S):
            k, v = pm.group(1), pm.group(2).strip()
            try:
                args[k] = json.loads(v)          # 值本身常是 JSON（列表/对象/数字）
            except json.JSONDecodeError:
                args[k] = v
        calls.append(_mk(name, args))
    return [c for c in calls if c["function"]["name"]]


def _mk(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:20], "type": "function",
            "function": {"name": name or "",
                         "arguments": json.dumps(args, ensure_ascii=False)}}


def _clean(text):
    """去掉思考段与工具调用块，留下给用户看的正文。"""
    t = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S)
    t = re.sub(r".*?</think>", "", t, flags=re.S)          # 模型会先写推理再给结论
    t = re.sub(r"<\|im_end\|>|<\|endoftext\|>", "", t)
    return t.strip()


@app.post("/v1/chat/completions")
def chat(req: ChatReq):
    tok, model = load()
    def _flat(c):
        """内容可能是字符串，也可能是 OpenAI 的多模态数组
        [{"type":"text","text":"..."}] —— Qwen 模板只吃字符串。"""
        if c is None:
            return ""
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(x.get("text", "") if isinstance(x, dict) else str(x)
                           for x in c)
        return str(c)

    # OpenAI 新协议把 system 改叫 developer，Qwen 的 chat 模板不认，
    # 直接抛 jinja2 TemplateError: Unexpected message role。
    ROLE_MAP = {"developer": "system"}

    msgs = []
    for m in req.messages:
        d = {"role": ROLE_MAP.get(m.role, m.role), "content": _flat(m.content)}
        if m.tool_calls:
            # OpenAI 协议里 arguments 是 JSON 字符串，Qwen 的模板却要字典：
            #     {%- for args_name, args_value in tool_call.arguments|items %}
            # 字符串进去直接 TypeError: Can only get item pairs from a mapping。
            # 多轮对话里 pi 会把上一轮的 tool_calls 发回来，所以这里必须转。
            fixed = []
            for c in m.tool_calls:
                c = dict(c)
                fn = dict(c.get("function") or {})
                a = fn.get("arguments")
                if isinstance(a, str):
                    try:
                        fn["arguments"] = json.loads(a) if a.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                elif not isinstance(a, dict):
                    fn["arguments"] = {}
                c["function"] = fn
                # 模板同时也认扁平写法 tool_call.name / tool_call.arguments
                c.setdefault("name", fn.get("name"))
                c["arguments"] = fn["arguments"]
                fixed.append(c)
            d["tool_calls"] = fixed
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        msgs.append(d)

    print("ROLES>>> %s" % [(m["role"], str(m.get("content"))[:40]) for m in msgs], flush=True)
    if req.tools:
        print("TOOLS>>> %s" % json.dumps(req.tools[:2], ensure_ascii=False)[:600], flush=True)
    def _fix_tool(t):
        """规整工具定义 —— Qwen 模板对 parameters 的形状很挑。

        pi 会把它全部内置工具一起发过来，其中有的 parameters 不是 mapping，
        模板里 .items() 直接抛 TypeError: Can only get item pairs from a mapping。
        缺 parameters 或类型不对的，补一个空 object schema。
        """
        if not isinstance(t, dict):
            return None
        fn = t.get("function") if t.get("type") == "function" else t
        if not isinstance(fn, dict) or not fn.get("name"):
            return None
        pr = fn.get("parameters")
        if not isinstance(pr, dict) or "properties" not in pr:
            pr = {"type": "object", "properties": {}, "required": []}
        if not isinstance(pr.get("properties"), dict):
            pr["properties"] = {}
        if not isinstance(pr.get("required"), list):
            pr["required"] = []
        pr.setdefault("type", "object")
        return {"type": "function",
                "function": {"name": fn["name"],
                             "description": fn.get("description", "") or "",
                             "parameters": pr}}

    kw = {"tokenize": False, "add_generation_prompt": True}
    if req.tools:
        clean = [x for x in (_fix_tool(t) for t in req.tools) if x]
        if clean:
            kw["tools"] = clean
    text = tok.apply_chat_template(msgs, **kw)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    t0 = time.time()
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=req.max_tokens,
                           do_sample=req.temperature > 0,
                           temperature=max(req.temperature, 1e-5),
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    raw = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)

    if os.environ.get("CLEARCHEM_DEBUG_RAW"):
        print("RAW>>> %r" % raw[:800], flush=True)
    calls = _parse_tool_calls(raw)
    content = _clean(raw)
    msg = {"role": "assistant", "content": content or None}
    if calls:
        msg["tool_calls"] = calls

    # pi 默认发流式请求，不支持就报 "Stream ended without finish_reason"。
    # 这里是伪流式：先整段生成完再按 SSE 吐出去。真流式要改 generate 的
    # streamer，收益是首字延迟，但 27B 单次生成本来就要几秒到几十秒，
    # 先把协议对上更要紧。
    if req.stream:
        cid = "chatcmpl-" + uuid.uuid4().hex[:20]
        created = int(time.time())

        def gen():
            head = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"},
                                 "finish_reason": None}]}
            yield "data: %s\n\n" % json.dumps(head, ensure_ascii=False)
            if content:
                for i in range(0, len(content), 40):
                    d = {"id": cid, "object": "chat.completion.chunk",
                         "created": created, "model": req.model,
                         "choices": [{"index": 0,
                                      "delta": {"content": content[i:i + 40]},
                                      "finish_reason": None}]}
                    yield "data: %s\n\n" % json.dumps(d, ensure_ascii=False)
            if calls:
                for i, c in enumerate(calls):
                    d = {"id": cid, "object": "chat.completion.chunk",
                         "created": created, "model": req.model,
                         "choices": [{"index": 0, "delta": {"tool_calls": [
                             {"index": i, "id": c["id"], "type": "function",
                              "function": c["function"]}]}, "finish_reason": None}]}
                    yield "data: %s\n\n" % json.dumps(d, ensure_ascii=False)
            tail = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {},
                                 "finish_reason": "tool_calls" if calls else "stop"}],
                    "usage": {"prompt_tokens": int(enc["input_ids"].shape[1]),
                              "completion_tokens": int(g.shape[1] - enc["input_ids"].shape[1]),
                              "total_tokens": int(g.shape[1])}}
            yield "data: %s\n\n" % json.dumps(tail, ensure_ascii=False)
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:20], "object": "chat.completion",
        "created": int(time.time()), "model": req.model,
        "choices": [{"index": 0, "message": msg,
                     "finish_reason": "tool_calls" if calls else "stop"}],
        "usage": {"prompt_tokens": int(enc["input_ids"].shape[1]),
                  "completion_tokens": int(g.shape[1] - enc["input_ids"].shape[1]),
                  "total_tokens": int(g.shape[1])},
        "_seconds": round(time.time() - t0, 1),
        "_raw": raw[:1200] if os.environ.get("CLEARCHEM_DEBUG_RAW") else None,
    }


if __name__ == "__main__":
    b, a = _paths()
    print("ChemQwen OpenAI 兼容接口 · 端口 %d · 模型名 %s" % (PORT, MODEL_ID))
    print("  底座 %s" % b)
    print("  适配器 %s" % a)
    print("  首次请求会载入 52 GB，约 160 秒")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
