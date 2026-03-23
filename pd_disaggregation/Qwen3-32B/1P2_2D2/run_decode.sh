#!/bin/sh
# Qwen3-32B/1P2_2D2：单脚本通过入参区分 Decode 实例。与 run_prefill.sh 配套。
# 用法：
#   bash run_decode.sh          # 后台启动 D0、D1 并 wait（供 sweep / start_decode 调用）
#   bash run_decode.sh 0        # 仅启动第 1 个 D（卡 2,3，端口 9010）
#   bash run_decode.sh 1        # 仅启动第 2 个 D（卡 4,5，端口 9011）
# 日志：vllm 写入 ${LOG_DIR}/decode_<rank>.log（LOG_DIR 默认 .）；非法入参的提示仍打 stderr。
# ========== 配置区（须与 run_prefill.sh 一致）==========
# NIC_NAME / LOCAL_IP 由 PdServiceCtl 注入；单独跑脚本时请 export，默认值与 pd_service_ctl 中常量一致。
nic_name="${NIC_NAME:-eth0}"
local_ip="${LOCAL_IP:-172.17.0.4}"
model_path="/root/autodl-tmp/models/Qwen3-32B"
transfer_engine_lib_path="/usr/local/lib"
python_lib_path="/root/.local/share/uv/python/cpython-3.11.15-linux-aarch64-gnu/lib"
dp_size=2
dp_ip="127.0.0.1"
dp_rpc_port=13495
# 每个 DP rank 对应：engine_port, visible_devices
# rank 0 -> 9010, 2,3；rank 1 -> 9011, 4,5
# ==========================================

run_one() {
    dp_rank=$1
    case $dp_rank in
        0) engine_port=9010; visible_devices="2,3" ;;
        1) engine_port=9011; visible_devices="4,5" ;;
        *) echo "Usage: $0 [0|1]" >&2; exit 1 ;;
    esac

    export ASCEND_RT_VISIBLE_DEVICES=$visible_devices

    if [ -f /usr/lib/aarch64-linux-gnu/libstdc++.so.6 ]; then
      export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libstdc++.so.6
    elif [ -f /usr/lib64/libstdc++.so.6 ]; then
      export LD_PRELOAD=/usr/lib64/libstdc++.so.6
    elif [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
      export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
    fi
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD
    export LD_LIBRARY_PATH="${python_lib_path:+$python_lib_path:}${transfer_engine_lib_path:+$transfer_engine_lib_path:}/usr/lib64:/usr/lib/aarch64-linux-gnu:/usr/lib:${LD_LIBRARY_PATH:-}"

    export HCCL_IF_IP=$local_ip
    export GLOO_SOCKET_IFNAME=$nic_name
    export TP_SOCKET_IFNAME=$nic_name
    export HCCL_SOCKET_IFNAME=$nic_name
    export OMP_PROC_BIND=false
    export OMP_NUM_THREADS=10
    export HCCL_BUFFSIZE=1024

    export VLLM_DP_SIZE=$dp_size
    export VLLM_DP_MASTER_IP=$dp_ip
    export VLLM_DP_MASTER_PORT=$dp_rpc_port
    export VLLM_DP_RANK_LOCAL=0
    export VLLM_DP_RANK=$dp_rank
    export VLLM_DP_SIZE_LOCAL=1

    export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
    export TASK_QUEUE_ENABLE=1
    export VLLM_WORKER_MULTIPROC_METHOD="fork"
    export VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED=1

    LOG_DIR="${LOG_DIR:-.}"
    mkdir -p "$LOG_DIR"
    exec >> "${LOG_DIR}/decode_${dp_rank}.log" 2>&1

    vllm serve "$model_path" \
        --host 0.0.0.0 \
        --port $engine_port \
        --tensor-parallel-size 2 \
        --nnodes 1 \
        --data-parallel-size $dp_size \
        --data-parallel-rank $dp_rank \
        --data-parallel-address $dp_ip \
        --data-parallel-rpc-port $dp_rpc_port \
        --data-parallel-size-local 1 \
        --seed 1024 \
        --served-model-name qwen3_32b \
        --dtype bfloat16 \
        --max-model-len 32768 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 256 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code \
        --gpu-memory-utilization 0.9 \
        --enforce-eager \
        --kv-transfer-config \
        '{
            "kv_connector": "MooncakeConnectorV1",
            "kv_buffer_device": "npu",
            "kv_role": "kv_consumer",
            "kv_parallel_size": "1",
            "kv_port": "20002",
            "engine_id": "1",
            "kv_connector_extra_config": {
                "prefill": { "dp_size": 1, "tp_size": 2 },
                "decode": { "dp_size": 2, "tp_size": 2 }
            },
            "kv_connector_module_path": "vllm_ascend.distributed.mooncake_connector"
        }'
}

if [ $# -eq 0 ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    LOG_DIR="${LOG_DIR:-.}"
    mkdir -p "$LOG_DIR"
    bash "$SCRIPT_DIR/run_decode.sh" 0 &
    bash "$SCRIPT_DIR/run_decode.sh" 1 &
    wait
else
    run_one "$1"
fi
