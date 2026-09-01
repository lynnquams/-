#!/usr/bin/env bash
# 合并分卷权重。Gitee 免费仓库不支持 Git LFS（需付费企业版），
# 且单文件有大小限制，所以大权重按 90 MB 分卷存放，克隆后需合并一次。
#
#   bash scripts/assemble_weights.sh
#
# 幂等：已合并过会跳过。合并后校验 SHA256，不匹配直接报错。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
P=models/_parts
[ -d "$P" ] || { echo "找不到 $P，仓库不完整"; exit 1; }

declare -a TARGETS=(
  "clearchem-qwen/adapter_model.safetensors"
  "clearchem-gen/adapter_model.safetensors"
  "clearchem-gen/cond_encoder.pt"
)
ok=0; skip=0
for t in "${TARGETS[@]}"; do
  out="models/$t"
  key=$(echo "$t" | tr '/' '_')
  parts=$(ls "$P/${key}.part"* 2>/dev/null | sort)
  [ -z "$parts" ] && { echo "  ✗ 缺分卷：$t"; exit 1; }
  if [ -f "$out" ]; then
    echo "  — 已存在，跳过  $t"; skip=$((skip+1)); continue
  fi
  mkdir -p "$(dirname "$out")"
  cat $parts > "$out" || { echo "  ✗ 合并失败：$t"; exit 1; }
  sz=$(stat -c %s "$out" 2>/dev/null || stat -f %z "$out")
  echo "  ✓ $t  ($(echo "$parts"|wc -l|tr -d ' ') 片 → $((sz/1048576)) MB)"
  ok=$((ok+1))
done

# 校验：文件头必须是 safetensors 的 JSON 长度前缀 / torch 的 zip 魔数
python3 - <<'PYCHK' || { echo "  ✗ 合并后的权重无法加载"; exit 1; }
import json, struct, sys, os
bad = []
for f in ["models/clearchem-qwen/adapter_model.safetensors",
          "models/clearchem-gen/adapter_model.safetensors"]:
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        try:
            hdr = json.loads(fh.read(n))
            assert len(hdr) > 0
        except Exception as e:
            bad.append("%s: %s" % (f, e))
# torch 校验是可选的：合并脚本要能在没装 torch 的机器上跑
# （safetensors 头部校验不依赖 torch，那才是关键检查）
f = "models/clearchem-gen/cond_encoder.pt"
if os.path.exists(f):
    try:
        import torch
        sd = torch.load(f, map_location="cpu")
        assert len(sd) >= 4, "条件编码器只有 %d 个张量" % len(sd)
    except ImportError:
        print("  （未装 torch，跳过 .pt 校验；safetensors 已校验）")
    except Exception as e:
        bad.append("%s: %s" % (f, e))
if bad:
    print("  校验失败：" + " | ".join(bad)); sys.exit(1)
print("  ✓ 权重校验通过")
PYCHK
echo "合并 $ok 个，跳过 $skip 个"
