#!/bin/bash
# 后台启动 Decode（PD 分离），供 run_sharegpt_sweep.sh 调用
TEST_NAME="${1:?Usage: $0 <test_name>}"
BATCH_SIZE="${2:?Usage: $0 <batch_size>}"

RUN_SCRIPT=/root/autodl-tmp/yyz/Qwen3-8B/run_decode_1card.sh
# 与 sharegpt.sh 一致：sweep 每轮会 export LOG_DIR=/root/autodl-tmp/yyz/log/${TEST_NAME}_${BATCH_SIZE}

PID_FILE="/tmp/vllm_decode.pid"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [decode] $*"
}

log "Starting decode in background (TEST_NAME=$TEST_NAME, BATCH_SIZE=$BATCH_SIZE)..."
source /root/autodl-tmp/py_venv/vllm2/bin/activate
nohup bash "$RUN_SCRIPT" \
    >> "$LOG_DIR/decode.log" 2>&1 &
echo $! > "$PID_FILE"
log "Decode started (PID $(cat "$PID_FILE")), log: $LOG_DIR/decode.log"
