TASK_NAME="${1:?Usage: $0 <TASK_NAME>}"
BATCH_SIZE="${2:?Usage: $0 <batch_size>}"
LOG_DIR="/root/autodl-tmp/pxy/share/workspace/logs/${TASK_NAME}"

source /root/autodl-tmp/py_venv/xy_tester/bin/activate
cd /root/autodl-tmp/pxy/share/workspace/benchmark

sed -i "s/batch_size=[0-9]*/batch_size=$BATCH_SIZE/g" ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py
ais_bench --models vllm_api_stream_chat_multiturn --datasets sharegpt_gen --mode perf --num-warmups 0 > $LOG_DIR/aisbench_${TASK_NAME}_bs${BATCH_SIZE}.log
