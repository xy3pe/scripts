TEST_NAME="${1:?Usage: $0 <test_name>}"
BATCH_SIZE="${2:?Usage: $0 <batch_size>}"


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/root/autodl-tmp/pxy/share/workspace/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SCRIPT_LOG="$LOG_DIR/startup_${TIMESTAMP}.log"
PID_FILE="/tmp/vllm_kv_both.pid"
TMP_CONFIG=$(mktemp /tmp/vllm_logging_XXXX.json)

mkdir -p "$LOG_DIR"

log() {
    msg="$(date '+%Y-%m-%d %H:%M:%S') $*"
    echo "$msg"
    echo "$msg" >> "$SCRIPT_LOG"
}

log "Test name     : $TEST_NAME"
log "Script log    : $SCRIPT_LOG"
log "batch size    : $BATCH_SIZE"

sed "s|vllm\.log|vllm_${TIMESTAMP}_${TEST_NAME}_${BATCH_SIZE}.log|" "$SCRIPT_DIR/logging.config" > "$TMP_CONFIG"
log "Logging config: $TMP_CONFIG -> $LOG_DIR/vllm_${TEST_NAME}.log"

export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
export MOONCAKE_CONFIG_PATH="/root/autodl-tmp/pxy/share/workspace/mooncake/mooncake.json"
export MODEL_PATH="/root/autodl-tmp/models"
# NPU buffer pool: quantity:size(MB)
# Allocates 4 buffers of 8MB each for KV transfer
export ASCEND_BUFFER_POOL=4:8
# Set the operator dispatch pipeline level to 1 and disable manual memory control in ACLGraph
export TASK_QUEUE_ENABLE=1

# [Optional] jemalloc
# jemalloc is for better performance, if `libjemalloc.so` is installed on your machine, you can turn it on.
# if os is Ubuntu
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
# if os is openEuler
# export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD
# Enable the AIVector core to directly schedule ROCE communication
export HCCL_OP_EXPANSION_MODE="AIV"
# Enable FlashComm_v1 optimization when tensor parallel is enabled.
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

export ASCEND_RT_VISIBLE_DEVICES=2,3

MODEL_NAME=Qwen3-8B

source /root/autodl-tmp/py_venv/vllm2/bin/activate

log "Starting vllm serve (model: $MODEL_NAME) ..."

VLLM_LOGGING_CONFIG_PATH="$TMP_CONFIG" \
vllm serve $MODEL_PATH/$MODEL_NAME \
    --dtype bfloat16 \
    --max-model-len 16k \
    --tensor-parallel-size 2 \
    --port 8131 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.9 \
    --enable-request-id-headers \
    --served-model-name Qwen3-8B \
    --kv-transfer-config '{
      "kv_connector": "MooncakeConnectorStoreV1",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
          "use_layerwise": false,
          "mooncake_rpc_port": "0",
          "load_async": true,
          "register_buffer": true
      }
  }' &

echo $! > "$PID_FILE"
log "vllm started (PID $(cat $PID_FILE))"
