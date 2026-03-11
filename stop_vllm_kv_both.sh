LOG_DIR="/root/autodl-tmp/pxy/share/workspace/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG="$LOG_DIR/shutdown_${TIMESTAMP}.log"
PID_FILE="/tmp/vllm_kv_both.pid"

mkdir -p "$LOG_DIR"

log() {
    msg="$(date '+%Y-%m-%d %H:%M:%S') $*"
    echo "$msg"
    echo "$msg" >> "$SCRIPT_LOG"
}

log "Script log: $SCRIPT_LOG"

if [ ! -f "$PID_FILE" ]; then
    log "No PID file found at $PID_FILE, vllm may not be running."
    exit 1
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    log "Process $PID is not running."
    rm -f "$PID_FILE"
    exit 1
fi

log "Sending SIGTERM to vllm process (PID $PID), waiting for exit..."
kill "$PID"

# Wait up to 30s for graceful shutdown, then force kill
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
log "vllm stopped."
