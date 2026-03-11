#!/bin/bash
# 停止 PD 分离模式下的 Decode 进程（应先于 Prefill 停止）
PID_FILE="/tmp/vllm_decode.pid"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [stop_decode] $*"
}

if [ ! -f "$PID_FILE" ]; then
    log "No PID file at $PID_FILE, decode may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
    log "Process $PID is not running."
    rm -f "$PID_FILE"
    exit 0
fi

log "Sending SIGTERM to decode (PID $PID), waiting for exit..."
kill "$PID"
i=0
while kill -0 "$PID" 2>/dev/null; do
    if [ $i -ge 30 ]; then
        log "Process did not exit in 30s, sending SIGKILL..."
        kill -9 "$PID"
        break
    fi
    sleep 1
    i=$((i + 1))
done
rm -f "$PID_FILE"
log "Decode stopped."
