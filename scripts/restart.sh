#!/bin/bash
# 重启服务：按端口找进程、确认死透、确认新的真起来了。
#
# 手动 kill 踩过四次坑：
#   ① kill 没生效，旧进程继续服务，改的代码不生效（看日志像是重启了）
#   ② 新进程因端口被占静默退出，日志停在"启动"那行
#   ③ kill -9 打断 SSH 连接
# 所以这里按端口精确定位、循环确认、最后验端口真的在新进程手里。
set -u
PORT="${1:-8900}"
CMD="${2:-scripts/serve.py}"
ROOT="${CLEARCHEM_ROOT:-/root/autodl-tmp/clearchem}"
LOG="${3:-/root/svc_$PORT.log}"

pids=$(ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
[ -z "$pids" ] && pids=$(pgrep -f "$CMD" 2>/dev/null)

for p in $pids; do
  kill "$p" 2>/dev/null
done
for i in $(seq 1 15); do
  sleep 2
  still=$(ss -lptnH "sport = :$PORT" 2>/dev/null | grep -c pid= || true)
  [ "${still:-0}" = "0" ] && break
done
still=$(ss -lptnH "sport = :$PORT" 2>/dev/null | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
if [ -n "$still" ]; then
  echo "端口 $PORT 仍被占用（pid $still），改用 -9"
  for p in $still; do kill -9 "$p" 2>/dev/null; done
  sleep 4
fi

cd "$ROOT" || { echo "进不去 $ROOT"; exit 1; }
PORT="$PORT" nohup python3 "$CMD" > "$LOG" 2>&1 &
newpid=$!
for i in $(seq 1 20); do
  sleep 2
  if ss -lptnH "sport = :$PORT" 2>/dev/null | grep -q "pid=$newpid"; then
    echo "已重启：$CMD 占用端口 $PORT，pid $newpid"
    exit 0
  fi
  kill -0 "$newpid" 2>/dev/null || { echo "新进程已退出，日志尾部："; tail -5 "$LOG"; exit 1; }
done
echo "40 秒内没等到端口 $PORT 被新进程占用，日志尾部："
tail -5 "$LOG"
exit 1
