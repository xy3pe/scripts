#!/bin/bash
set -euo pipefail

TEST_NAME="${1:?Usage: $0 <test_name>}"

BATCH_SIZES=(35 30 25 20 10 5)
VLLM_PORT=8131
VLLM_READY_TIMEOUT=300   # 最多等待 5 分钟

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_SCRIPTS_DIR="/root/autodl-tmp/pxy/share/workspace/scripts"
LOG_DIR="/root/autodl-tmp/pxy/share/workspace/logs"
SWEEP_LOG="$LOG_DIR/sweep_${TEST_NAME}.log"

mkdir -p "$LOG_DIR"

# ── 日志函数 ────────────────────────────────────────────────────────────────
log() {
    local msg
    msg="$(date '+%Y-%m-%d %H:%M:%S') [sweep] $*"
    echo "$msg" | tee -a "$SWEEP_LOG"
}

# ── 等待 vllm 健康接口就绪 ──────────────────────────────────────────────────
wait_for_vllm() {
    log "Waiting for vllm on port $VLLM_PORT (timeout ${VLLM_READY_TIMEOUT}s)..."
    local elapsed=0
    while ! curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
        if [ "$elapsed" -ge "$VLLM_READY_TIMEOUT" ]; then
            log "ERROR: vllm did not become ready within ${VLLM_READY_TIMEOUT}s."
            return 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    log "vllm is ready (waited ${elapsed}s)."
}

# ── 停止 vllm（用于 trap，忽略错误）────────────────────────────────────────
stop_vllm() {
    log "Stopping vllm..."
    bash "$VLLM_SCRIPTS_DIR/stop_vllm_kv_both.sh" || true
}

# 异常退出时确保 vllm 被停止
trap 'log "Interrupted, cleaning up..."; stop_vllm; exit 1' INT TERM

# ── 主循环 ──────────────────────────────────────────────────────────────────
log "Sweep start: TEST_NAME=$TEST_NAME  BATCH_SIZES=${BATCH_SIZES[*]}"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    log "========== Round: BATCH_SIZE=$BATCH_SIZE =========="

    # 1. 启动 vllm
    log "Starting vllm (TEST_NAME=${TEST_NAME}_bs${BATCH_SIZE})..."
    bash "$VLLM_SCRIPTS_DIR/start_vllm_kv_both.sh" "${TEST_NAME}_bs${BATCH_SIZE}" "$BATCH_SIZE"

    # 2. 等待 vllm 就绪，失败则停止并跳过本轮
    if ! wait_for_vllm; then
        log "ERROR: skipping BATCH_SIZE=$BATCH_SIZE due to vllm startup failure."
        stop_vllm
        continue
    fi

    # 3. 运行压测
    log "Running sharegpt benchmark (BATCH_SIZE=$BATCH_SIZE)..."
    if bash "$SCRIPT_DIR/sharegpt.sh" "$TEST_NAME" "$BATCH_SIZE"; then
        log "Benchmark finished: BATCH_SIZE=$BATCH_SIZE"
    else
        log "WARNING: benchmark exited with error for BATCH_SIZE=$BATCH_SIZE, continuing..."
    fi

    # 4. 停止 vllm，为下一轮做准备
    stop_vllm
    log "vllm stopped. Waiting 5s before next round..."
    sleep 5
done

log "All rounds completed. Logs: $LOG_DIR"
