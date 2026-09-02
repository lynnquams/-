/**
 * ClearChem 接入 pi：电解液分子性质、成膜添加剂筛选、配方推荐、化学问答。
 *
 * 这层只做转发。真正的计算在远端 —— 两个模型（ChemQwen 27B、ClearChem-Gen）
 * 加五个引擎（五把尺子、电导率尺子、xTB、MACE、ABACUS），一百多 GB 权重，
 * 本机装不下也不该装。
 *
 * 每个工具的描述里都写了实测边界，而且写的是"什么时候别用它"。
 * 原因是性质尺子在电解液分子上会给出方向相反的答案（电化学排序 2/6，
 * 比随机的 3/6 还差，把 VC/FEC 判成比 EC 更难还原），而它不会报错。
 * 模型看不到这条边界，就会拿它去筛添加剂，然后给出反的结论。
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const URL = (process.env.CLEARCHEM_URL ?? "http://localhost:8900").replace(/\/$/, "");
const TIMEOUT = Number(process.env.CLEARCHEM_TIMEOUT ?? 900_000);

function textResult(obj: unknown, note?: string) {
  const body = typeof obj === "string" ? obj : JSON.stringify(obj, null, 1);
  return {
    content: [{ type: "text" as const, text: note ? `${note}\n${body}` : body }],
    details: typeof obj === "object" && obj !== null ? (obj as object) : {},
  };
}

/** 调远端 ClearChem。连不上时把修法一起返回，不然模型只知道"失败了"。 */
async function api(path: string, payload?: unknown, signal?: AbortSignal) {
  try {
    const r = await fetch(`${URL}/${path}`, {
      method: payload === undefined ? "GET" : "POST",
      headers: payload === undefined ? {} : { "Content-Type": "application/json" },
      body: payload === undefined ? undefined : JSON.stringify(payload),
      signal: signal ?? AbortSignal.timeout(TIMEOUT),
    });
    if (!r.ok) return { error: `ClearChem 返回 ${r.status}: ${(await r.text()).slice(0, 500)}` };
    return await r.json();
  } catch (e) {
    return {
      error:
        `连不上 ClearChem（${URL}）：${e instanceof Error ? e.message : String(e)}\n` +
        `服务没起：ssh <服务器> 'cd .../clearchem && python3 scripts/serve.py'\n` +
        `端口没转发：ssh -N -L 8900:localhost:8900 <服务器>`,
    };
  }
}

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "clearchem_health",
    label: "查服务",
    description:
      "查 ClearChem 服务状态：两个模型载入了没、是不是轻量档、跑了多久。" +
      "任何一次会话里第一次用 ClearChem 之前先查一眼 —— 服务没起时" +
      "后面每个调用都会失败，而失败信息看起来像是分子有问题。",
    promptSnippet: "查 ClearChem 服务是否就绪",
    parameters: Type.Object({}),
    async execute(_id, _p, signal) {
      return textResult(await api("health", undefined, signal));
    },
  });

  pi.registerTool({
    name: "clearchem_orbitals",
    label: "轨道能级",
    description:
      "电解液分子的 HOMO / LUMO / 带隙，走 GFN2-xTB，约 1 秒一个分子。" +
      "碳酸酯、醚、砜、腈这些电解液分子一律用这个。" +
      "实测：同一套电化学排序检验 6/6 通过，" +
      "LUMO(EC)−LUMO(VC)=+0.372 eV、−LUMO(FEC)=+0.566 eV，与文献 0.3~0.9 吻合。" +
      "边界：气相单构象，溶剂化会整体平移能级，所以只能做同类分子之间的相对比较，" +
      "不能当绝对电位。",
    promptSnippet: "用 xTB 算电解液分子的 HOMO/LUMO/带隙",
    promptGuidelines: [
      "电解液分子（碳酸酯/醚/砜/腈）问性质，用 clearchem_orbitals，不要用 clearchem_predict。",
    ],
    parameters: Type.Object({
      smiles: Type.Array(Type.String(), { description: "SMILES 列表" }),
    }),
    async execute(_id, params, signal) {
      return textResult(await api("orbitals", { smiles: params.smiles }, signal));
    },
  });

  pi.registerTool({
    name: "clearchem_screen_additive",
    label: "筛成膜添加剂",
    description:
      "判断一个分子能不能做 SEI 成膜添加剂：算它的 LUMO 比参照溶剂（默认 EC）低多少。" +
      "低得越多越容易先于溶剂还原、在负极表面成膜。" +
      "参照点是实测的：VC 低 0.372 eV、FEC 低 0.566 eV。低 0.2 eV 以上判为可成膜。",
    promptSnippet: "判断分子能否做 SEI 成膜添加剂",
    parameters: Type.Object({
      smiles: Type.Array(Type.String(), { description: "候选分子 SMILES" }),
      reference: Type.Optional(
        Type.String({ description: "参照溶剂 SMILES，默认 EC（C1COC(=O)O1）" }),
      ),
    }),
    async execute(_id, params, signal) {
      return textResult(
        await api("screen_additive",
          { smiles: params.smiles, reference: params.reference ?? "C1COC(=O)O1" }, signal),
      );
    },
  });

  pi.registerTool({
    name: "clearchem_predict",
    label: "性质尺子（通用小分子）",
    description:
      "五把性质尺子（gap/homo/lumo/ip/ea），毫秒级，用于通用小分子粗筛。" +
      "⚠ 电解液分子上不可用，会给出方向相反的答案：" +
      "碳酸酯 LUMO 的真实跨度是 1.17 eV，尺子把它压成 0.27 eV，" +
      "而尺子自身的测试 MAE 就是 0.28 eV —— 要分辨的差距比工具误差还小。" +
      "实测电化学排序只有 2/6（随机是 3/6），把 VC 和 FEC 判成比 EC 更难还原，" +
      "与它们做成膜添加剂的机理完全相反。碳酸酯/醚/砜请改用 clearchem_orbitals。",
    promptSnippet: "用训练出来的尺子预测通用小分子性质（电解液分子勿用）",
    parameters: Type.Object({
      smiles: Type.Array(Type.String()),
    }),
    async execute(_id, params, signal) {
      const r = await api("predict", { smiles: params.smiles }, signal);
      return textResult(r, "提醒：若这些是电解液分子（碳酸酯/醚/砜/腈），本工具的排序不可信，改用 clearchem_orbitals。");
    },
  });

  pi.registerTool({
    name: "clearchem_design_molecule",
    label: "条件分子生成",
    description:
      "给定目标性质生成候选分子，走 ClearChem-Gen（ether0 底座 + 条件编码器）。" +
      "实测条件遵循 MAE 0.109 eV，六个目标点的偏差全在 ±0.03 以内；" +
      "Novelty 0.928、Validity 0.992、合成可及性 SA 3.40。",
    promptSnippet: "按目标性质生成候选分子",
    parameters: Type.Object({
      targets: Type.Record(Type.String(), Type.Number(), {
        description: '目标性质，如 {"gap": 8.5}',
      }),
      n: Type.Optional(Type.Number({ description: "生成个数，默认 10" })),
    }),
    async execute(_id, params, signal) {
      return textResult(
        await api("design_molecule", { targets: params.targets, n: params.n ?? 10 }, signal),
      );
    },
  });

  pi.registerTool({
    name: "clearchem_design_formulation",
    label: "配方推荐",
    description:
      "按目标电导率推荐电解液配方（锂盐 + 溶剂 + 浓度）。" +
      "⚠ 只在 CALiSol 覆盖的 14 种锂盐 × 38 种溶剂范围内可信：" +
      "按文献切分的 5 折交叉验证 R² 全部 ≤0.29，其中三折为负 —— " +
      "换到没见过的溶剂体系，它给的是瞎猜。新体系请走 clearchem_simulate_conductivity。",
    promptSnippet: "按目标电导率推荐电解液配方",
    parameters: Type.Object({
      k_target: Type.Number({ description: "目标电导率 mS/cm" }),
      n: Type.Optional(Type.Number({ description: "候选个数，默认 8" })),
      T: Type.Optional(Type.Number({ description: "温度 K，默认 298.15" })),
      salt: Type.Optional(Type.String({ description: "指定锂盐，留空则不限" })),
    }),
    async execute(_id, params, signal) {
      return textResult(
        await api("design_formulation", {
          k_target: params.k_target, n: params.n ?? 8,
          T: params.T ?? 298.15, salt: params.salt ?? "",
        }, signal),
      );
    },
  });

  pi.registerTool({
    name: "clearchem_simulate_conductivity",
    label: "分子动力学算电导率",
    description:
      "用 MACE-MP-0b2 分子动力学算电解液电导率。不依赖实验数据，" +
      "因而能算尺子外推不了的新体系。代价约 3.4 小时一个配方。" +
      "⚠ 不能用于配方排序：同一体系换随机种子，实测扩散系数相差 1.84~2.20 倍，" +
      "而 EC:DMC 与 PC 的真实电导率差距只有 1.72 倍 —— " +
      "自身波动大于要分辨的差距，实测排序会翻转。单次结果只能读数量级。" +
      "要能排序需每配方 5~8 个种子或把离子数提到 24 以上，代价 17~27 小时。",
    promptSnippet: "用分子动力学算电解液电导率（耗时数小时）",
    promptGuidelines: [
      "这是小时级的作业，调用前先跟用户确认值不值得等。",
      "不要用单次 MD 结果比较两个配方的高低，实测会翻转。",
    ],
    parameters: Type.Object({
      comp: Type.Record(Type.String(), Type.Number(), {
        description: '溶剂组成，如 {"EC": 10, "DMC": 10}',
      }),
      rho: Type.Optional(Type.Number({ description: "目标密度 g/cm³，默认 1.20" })),
      n_ion: Type.Optional(Type.Number({ description: "Li+ 个数，默认 6" })),
      ps: Type.Optional(Type.Number({ description: "轨迹长度 ps，默认 1000" })),
    }),
    async execute(_id, params, signal) {
      return textResult(
        await api("simulate_conductivity", {
          comp: params.comp, rho: params.rho ?? 1.2,
          n_ion: params.n_ion ?? 6, ps: params.ps ?? 1000,
        }, signal),
      );
    },
  });

  pi.registerTool({
    name: "clearchem_ask",
    label: "化学问答",
    description:
      "问 ChemQwen（Qwen3.8-27B + LoRA 化学特调）。" +
      "ChemBench 官方口径 2,785 题全量：接 Python 工具 0.6445、纯模型 0.6316，" +
      "裸底座是 0.5964。tool=true 时数值题会让模型写 Python 现算，" +
      "实测数值题正确率从 0.46 提到 0.60。",
    promptSnippet: "问化学特调模型",
    parameters: Type.Object({
      question: Type.String(),
      tool: Type.Optional(Type.Boolean({ description: "数值题是否允许写 Python 现算" })),
    }),
    async execute(_id, params, signal) {
      return textResult(
        await api("ask", { question: params.question, tool: params.tool ?? false }, signal),
      );
    },
  });

  pi.registerCommand("clearchem", {
    description: "电解液计算：轨道能级 / 成膜添加剂 / 配方推荐 / 化学问答",
    handler: async (args, ctx) => {
      const r = await api("health");
      if (r.error) {
        ctx.ui.notify(`ClearChem 不可用: ${r.error.split("\n")[0]}`, "error");
        return;
      }
      const mode = r.light ? "轻量档（未载两个大模型）" : "完整档";
      ctx.ui.notify(
        `ClearChem 就绪 · ${mode} · 已运行 ${Math.round(r.uptime_s)}s` +
          (args ? ` · 参数: ${args}` : ""),
        "info",
      );
    },
  });
}
