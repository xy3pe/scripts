TEST_NAME="${1:?Usage: $0 <test_name>}"
BATCH_SIZE="${2:?Usage: $0 <batch_size>}"
LOG_DIR="/root/autodl-tmp/yyz/log/${TEST_NAME}_${BATCH_SIZE}"

source /root/autodl-tmp/py_venv/tester/bin/activate
cd /root/autodl-tmp/code/benchmark

CONFIG_PY="ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py"
sed -i "s/batch_size=[0-9]*/batch_size=$BATCH_SIZE/g" "$CONFIG_PY"
ais_bench --models vllm_api_stream_chat_multiturn --datasets sharegpt_gen --mode perf > "$LOG_DIR/aisbench.log"