#!/usr/bin/env bash
# ClearChem 一键部署 —— 目标是在任何一台机器上都能跑起来。
#
#   bash scripts/deploy.sh                完整（含底座，需 GPU + 80GB 盘）
#   bash scripts/deploy.sh --lite         轻量（不下底座，只跑性质预测+配方推荐，CPU 即可）
#   bash scripts/deploy.sh --no-base      装环境和本仓库权重，底座自备
#   bash scripts/deploy.sh --check        只做环境自检
#
# 分级能力（按机器条件自动降级，缺什么禁用什么，其余照常работа）：
#   任何机器（含纯 CPU、无网）  性质预测 · 配方推荐          需 ~2 GB
#   GPU ≥ 24 GB                + 分子生成                    需 ~60 GB
#   GPU ≥ 60 GB                + 知识问答（27B）             需 ~120 GB
set -uo pipefail

ROOT="${CLEARCHEM_ROOT:-$HOME/clearchem}"
PY="${PYTHON:-python3}"
MODE="${1:-full}"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"
REPO="https://gitee.com/lin-fangyue/computing-platform.git"

c() { printf '\033[1;36m[deploy]\033[0m %s\n' "$*"; }
w() { printf '\033[1;33m[跳过]\033[0m %s\n' "$*"; }
e() { printf '\033[1;31m[错误]\033[0m %s\n' "$*" >&2; }
die() { e "$*"; exit 1; }

# ---------- 0. 环境探测（不假设任何东西）----------
c "探测运行环境"
command -v "$PY" >/dev/null 2>&1 || { for p in python3.12 python3.11 python3.10 python; do
    command -v $p >/dev/null 2>&1 && PY=$p && break; done; }
command -v "$PY" >/dev/null 2>&1 || die "找不到 python3（试过 python3/3.12/3.11/3.10/python）"
PYVER=$($PY -c 'import sys;print("%d.%d"%sys.version_info[:2])')
c "  Python $PYVER ($PY)"
$PY -c 'import sys; sys.exit(0 if sys.version_info>=(3,9) else 1)' || die "需要 Python ≥3.9，当前 $PYVER"

HAS_GPU=0; GPU_MEM=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_GPU=1
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null|sort -rn|head -1)
  GPU_MEM=${GPU_MEM:-0}
  c "  GPU $(nvidia-smi --query-gpu=name --format=csv,noheader|head -1) · ${GPU_MEM} MiB"
else
  w "  未检测到 GPU，将以 CPU 模式部署（性质预测和配方推荐可用；生成器不可用）"
fi

FREE_GB=$(df -Pk "$(dirname "$ROOT")" 2>/dev/null | tail -1 | awk '{print int($4/1048576)}')
FREE_GB=${FREE_GB:-0}
c "  可用磁盘 ${FREE_GB} GB"

# 网络探测：决定用不用镜像、能不能下底座
NET_OK=0
for u in "$MIRROR" "https://gitee.com"; do
  if curl -sf -m 8 -o /dev/null "$u" 2>/dev/null; then NET_OK=1; break; fi
done
[ "$NET_OK" = 1 ] && c "  外网可达" || w "  外网不可达，只能用本地已有资源"

# 按条件决定装什么
WANT_BASE=1; WANT_GEN=1
[ "$MODE" = "--lite" ] && WANT_BASE=0 && WANT_GEN=0
[ "$MODE" = "--no-base" ] && WANT_BASE=0
[ "$HAS_GPU" = 0 ] && WANT_GEN=0 && WANT_BASE=0 && w "  无 GPU → 自动切轻量模式"
[ "$HAS_GPU" = 1 ] && [ "$GPU_MEM" -lt 20000 ] && WANT_GEN=0 && WANT_BASE=0 \
  && w "  显存 ${GPU_MEM}MiB <20GB → 自动切轻量模式"
[ "$WANT_BASE" = 1 ] && [ "$FREE_GB" -lt 80 ] && WANT_BASE=0 \
  && w "  磁盘 ${FREE_GB}GB <80GB → 不下载底座"
[ "$NET_OK" = 0 ] && WANT_BASE=0

# ---------- 1. Python 依赖 ----------
c "安装依赖"
PIPFLAGS="--quiet --disable-pip-version-check"
[ "$NET_OK" = 1 ] && PIPIDX="-i https://pypi.tuna.tsinghua.edu.cn/simple" || PIPIDX=""
$PY -m pip install $PIPFLAGS --upgrade pip $PIPIDX 2>/dev/null
CORE="numpy pandas scipy rdkit"
[ "$WANT_GEN" = 1 ] && CORE="$CORE torch transformers peft accelerate safetensors"
[ "$WANT_GEN" = 0 ] && CORE="$CORE torch"     # 打分器也要 torch，但 CPU 版即可
for pkg in $CORE; do
  $PY -c "import ${pkg/-/_}" 2>/dev/null && continue
  c "  装 $pkg"
  $PY -m pip install $PIPFLAGS $PIPIDX "$pkg" 2>&1|tail -1 || e "  $pkg 安装失败"
done
$PY - <<'PYCHK' || die "依赖自检未通过，请手动安装 numpy pandas scipy rdkit torch"
import numpy, pandas, scipy, torch
from rdkit import Chem
assert Chem.MolFromSmiles("CCO") is not None
print("  核心依赖就绪 · torch %s · CUDA %s"
      % (torch.__version__, torch.cuda.is_available()))
PYCHK

# ---------- 2. 仓库与权重 ----------
if [ ! -d "$ROOT/.git" ]; then
  [ "$NET_OK" = 0 ] && die "无网且 $ROOT 不存在，请先手动放置仓库"
  c "克隆仓库 → $ROOT"
  git clone "$REPO" "$ROOT" || die "克隆失败"
else
  c "仓库已存在，尝试更新"
  git -C "$ROOT" pull --ff-only 2>/dev/null || w "  更新失败，用现有版本"
fi
cd "$ROOT" || die "进不去 $ROOT"

# 大权重按 90 MB 分卷存放（Gitee 免费仓库不支持 LFS），克隆后合并一次
c "合并分卷权重"
bash scripts/assemble_weights.sh || die "权重合并失败"

# ---------- 3. 底座（可选）----------
if [ "$WANT_BASE" = 1 ]; then
  c "下载底座（镜像 $MIRROR，约 60 GB，可中断续传）"
  export HF_ENDPOINT="$MIRROR"
  $PY -m pip install $PIPFLAGS $PIPIDX huggingface_hub 2>/dev/null
  CLEARCHEM_ROOT="$ROOT" $PY - <<'PYDL'
import os
from huggingface_hub import snapshot_download
root = os.environ["CLEARCHEM_ROOT"]
for repo, sub in [("futurehouse/ether0", "bases/ether0")]:
    tgt = os.path.join(root, sub)
    if os.path.exists(os.path.join(tgt, "config.json")):
        print("  %s 已存在" % sub); continue
    try:
        snapshot_download(repo_id=repo, local_dir=tgt, max_workers=4, resume_download=True)
        print("  %s 完成" % sub)
    except Exception as ex:
        print("  ★ %s 失败：%s" % (repo, str(ex)[:80]))
PYDL
else
  w "不下载底座 → 分子生成与知识问答不可用；性质预测和配方推荐照常"
fi

# ---------- 4. 配方数据 ----------
mkdir -p "$ROOT/data"
if [ ! -f "$ROOT/data/CALiSol-23.csv" ] && [ "$NET_OK" = 1 ]; then
  c "获取 CALiSol-23 配方数据"
  rm -rf /tmp/_cal && git clone --depth 1 https://github.com/Pele0599/CALiSol-23.git /tmp/_cal 2>/dev/null \
    && cp "/tmp/_cal/CALiSol-23 Dataset.csv" "$ROOT/data/CALiSol-23.csv" && rm -rf /tmp/_cal \
    && c "  $(wc -l < "$ROOT/data/CALiSol-23.csv") 行" || w "  获取失败，配方层不可用"
fi

# ---------- 5. 配置 ----------
DEV=cpu; [ "$HAS_GPU" = 1 ] && DEV=cuda
cat > "$ROOT/clearchem/config.json" <<CFG
{
  "root": "$ROOT",
  "device": "$DEV",
  "bases": {"ether0": "$ROOT/bases/ether0", "qwen": "$ROOT/bases/qwen"},
  "adapters": {"generator": "$ROOT/models/clearchem-gen", "qwen": "$ROOT/models/clearchem-qwen"},
  "scorers": "$ROOT/models/scorers",
  "calisol": "$ROOT/data/CALiSol-23.csv"
}
CFG
c "配置已写入"

# ---------- 6. 自检 ----------
if [ "$MODE" != "--skip-check" ]; then
  c "部署自检（真实调用，不是导入测试）"
  cd "$ROOT" && $PY -m clearchem.selfcheck
  RC=$?
  [ $RC -ne 0 ] && die "自检未通过（退出码 $RC）"
fi

echo
c "部署完成 → $ROOT"
echo
echo "    cd $ROOT && $PY"
echo "    >>> from clearchem import ClearChem"
echo "    >>> cc = ClearChem(load_generator=$([ "$WANT_GEN" = 1 ] && echo True || echo False))"
echo "    >>> cc.predict(['C1COC(=O)O1'])"
echo "    >>> cc.design_formulation(k_target=10.0, n=5)"
[ "$WANT_GEN" = 1 ] && echo "    >>> cc.design_molecule({'gap': 8.5}, n=5)"
echo
