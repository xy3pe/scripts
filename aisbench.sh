
TASK_NAME="${1:?Usage: $0 <TASK_NAME>}"
BATCH_SIZE="${2:?Usage: $0 <batch_size>}"
LOG_DIR="/root/autodl-tmp/pxy/share/workspace/logs"

source /root/autodl-tmp/py_venv/tester/bin/activate
cd /root/autodl-tmp/pxy/share/workspace/benchmark

sed -i "s/batch_size=[0-9]*/batch_size=$BATCH_SIZE/g" ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py
ais_bench --models vllm_api_stream_chat_multiturn --datasets sharegpt_gen --mode perf > $LOG_DIR/${TASK_NAME}/aisbench_${TASK_NAME}_bs${BATCH_SIZE}.log