#!/bin/bash
set -euo pipefail

# PD 分离模式：每轮先起 Prefill，再起 Decode，压测性能，最后先停 Decode 再停 Prefill
TEST_NAME="${1:?Usage: $0 <test_name>}"

BATCH_SIZES=(35 30 25 20 15 10 5)
VLLM_PORT=9010
VLLM_PREFILL_PORT=9000   # Prefill 健康检查（run_prefill_1card.sh 中 engine_port=9000）
VLLM_READY_TIMEOUT=300   # 最多等待 5 分钟

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/root/autodl-tmp/yyz/log"
SWEEP_LOG="$LOG_DIR/sweep_${TEST_NAME}.log"
export LOG_DIR

mkdir -p "$LOG_DIR"

# ── 日志函数 ────────────────────────────────────────────────────────────────
log() {
    local msg
    msg="$(date '+%Y-%m-%d %H:%M:%S') [sweep] $*"
    echo "$msg" | tee -a "$SWEEP_LOG"
}

# ── 等待某端口健康接口就绪 ───────────────────────────────────────────────────
wait_for_port() {
    local port=$1
    local name=${2:-$port}
    log "Waiting for $name on port $port (timeout ${VLLM_READY_TIMEOUT}s)..."
    local elapsed=0
    while ! curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; do
        if [ "$elapsed" -ge "$VLLM_READY_TIMEOUT" ]; then
            log "ERROR: $name did not become ready within ${VLLM_READY_TIMEOUT}s."
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "$name is ready (waited ${elapsed}s)."
}

# ── 停止 PD 双进程（先 Decode 后 Prefill）───────────────────────────────────
stop_vllm() {
    log "Stopping vllm (PD: decode then prefill)..."
    bash "$SCRIPT_DIR/stop_decode_1card.sh" || true
    bash "$SCRIPT_DIR/stop_prefill_1card.sh" || true
}

# 异常退出时确保进程被停止
trap 'log "Interrupted, cleaning up..."; stop_vllm; exit 1' INT TERM

# ── 主循环 ──────────────────────────────────────────────────────────────────
log "Sweep start (PD mode): TEST_NAME=$TEST_NAME  BATCH_SIZES=${BATCH_SIZES[*]}"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    log "========== Round: BATCH_SIZE=$BATCH_SIZE =========="
    export LOG_DIR="/root/autodl-tmp/yyz/log/${TEST_NAME}_${BATCH_SIZE}"
    mkdir -p "$LOG_DIR"

    # 1. 启动 Prefill
    log "Starting prefill (TEST_NAME=${TEST_NAME}_bs${BATCH_SIZE})..."
    bash "$SCRIPT_DIR/start_prefill_1card.sh" "${TEST_NAME}_bs${BATCH_SIZE}" "$BATCH_SIZE"

    # 2. 等待 Prefill 就绪
    if ! wait_for_port "$VLLM_PREFILL_PORT" "prefill"; then
        log "ERROR: skipping BATCH_SIZE=$BATCH_SIZE due to prefill startup failure."
        stop_vllm
        continue
    fi

    # 3. 启动 Decode
    log "Starting decode (TEST_NAME=${TEST_NAME}_bs${BATCH_SIZE})..."
    bash "$SCRIPT_DIR/start_decode_1card.sh" "${TEST_NAME}_bs${BATCH_SIZE}" "$BATCH_SIZE"

    # 4. 等待 Decode 就绪（对外 API）
    if ! wait_for_port "$VLLM_PORT" "decode"; then
        log "ERROR: skipping BATCH_SIZE=$BATCH_SIZE due to decode startup failure."
        stop_vllm
        continue
    fi

    # 5. 运行压测（连 Decode 端口）
    log "Running sharegpt benchmark (BATCH_SIZE=$BATCH_SIZE..."
    if bash "$SCRIPT_DIR/sharegpt.sh" "$TEST_NAME" "$BATCH_SIZE"; then
        log "Benchmark finished: BATCH_SIZE=$BATCH_SIZE"
    else
        log "WARNING: benchmark exited with error for BATCH_SIZE=$BATCH_SIZE, continuing..."
    fi

    # 6. 停止 PD，为下一轮做准备
    stop_vllm
    log "vllm (PD) stopped. Waiting 5s before next round..."
    sleep 5
done

log "All rounds completed. Logs: $LOG_DIR"
