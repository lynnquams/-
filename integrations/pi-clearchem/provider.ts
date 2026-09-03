/**
 * 把 ChemQwen 注册成 pi 的对话模型。
 *
 * 为什么值得这么做：ChemQwen 是完整的 Qwen3.5-27B 加化学 LoRA，
 * 工具调用能力实测完好 —— 算 LUMO 会调 clearchem_orbitals、
 * 说"生成分子"会调 clearchem_design_molecule、问水的分子式则直接答不乱调。
 * 化学微调只加强了化学，没削弱通用能力。
 *
 * 于是整条链可以全是自己的东西，不依赖任何外部 API：
 *     pi ──▶ ChemQwen（对话 + 工具决策） ──▶ ClearChem 九个工具
 *
 * 前提：远端起了 OpenAI 兼容接口并做了端口转发
 *     ssh <服务器> 'cd .../clearchem && python3 scripts/openai_api.py'
 *     ssh -N -L 8901:localhost:8901 <服务器>
 */
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const BASE = process.env.CHEMQWEN_URL ?? "http://localhost:8901/v1";

export default function (pi: ExtensionAPI) {
  pi.registerProvider("clearchem", {
    name: "ClearChem (ChemQwen)",
    baseUrl: BASE,
    apiKey: "local",          // 本地接口不校验，占位即可
    api: "openai-completions",
    models: [
      {
        id: "qwen-chem",
        name: "ChemQwen 27B (化学特调)",
        reasoning: true,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 32768,
        maxTokens: 4096,
      },
    ],
  });
}
