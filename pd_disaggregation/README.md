# PD 分离（Prefill / Decode）编排与结果分析

本目录包含 **PD 分离推理** 的启停编排以及压测日志的离线分析脚本。

所有部署配置集中在 **`configs/`** 目录下的 YAML 文件中，每个文件描述一个完整的 PD 集群（模型路径、实例列表、端口、卡号等），无需手写 shell 脚本。

---

## 快速开始

```bash
# 一键拉起 PD 集群（prefill → decode → 代理）
python pd_service_ctl.py start --config configs/qwen3_32b_1p2_2d2.yaml

# 预览将要执行的命令（不实际启动）
python pd_service_ctl.py start --config configs/qwen3_32b_1p2_2d2.yaml --dry_run

# 停止所有实例
python pd_service_ctl.py stop --config configs/qwen3_32b_1p2_2d2.yaml
# 或扫描 /tmp/vllm_*.pid 全部停止（无需指定配置）
python pd_service_ctl.py stop
```

---

## 配置文件（`configs/`）

每个 YAML 文件定义一个 PD 集群，核心结构：

```yaml
cluster_name: "Qwen3-32B_1P2_2D2"

model:
  path: "/root/autodl-tmp/models/Qwen3-32B"
  served_name: "qwen3_32b"

venv:
  vllm: "/root/autodl-tmp/py_venv/vllm2"

network:
  nic_name: null    # null = 自动探测
  local_ip: null

vllm_defaults:
  dtype: "bfloat16"
  max_model_len: 32768
  # ... 所有实例共享的 vLLM serve 参数

kv_connector:
  type: "MooncakeConnectorV1"
  buffer_device: "npu"
  module_path: "vllm_ascend.distributed.mooncake_connector"

prefill:
  - name: "P0"
    port: 9000
    devices: "0,1"
    tensor_parallel_size: 2
    # ...

decode:
  - name: "D0"
    port: 9010
    devices: "2,3"
    tensor_parallel_size: 2
    # ...

proxy:
  port: 8000    # 设为 null 或删除此段则不启代理
```

**自动推算**：`kv_connector_extra_config` 中的 `prefill.dp_size` / `decode.dp_size` 由实例列表长度自动计算，无需手动填写。

**现有配置**：

| 文件 | 说明 |
|------|------|
| `qwen3_32b_1p2_2d2.yaml` | Qwen3-32B, 1P(TP=2) + 2D(TP=2)，共 6 卡 |
| `qwen3_32b_1p2_1d2.yaml` | Qwen3-32B, 1P(TP=2) + 1D(TP=2)，共 4 卡 |
| `qwen3_8b_1p2_2d2.yaml` | Qwen3-8B, 1P(TP=2) + 2D(TP=2)，共 6 卡 |

---

## `pd_service_ctl.py`

**作用**：统一拉起 / 停止 **Prefill → Decode →（可选）代理**。

- **命令行**
  - `python pd_service_ctl.py start --config <yaml> [--log_dir …] [--no_wait] [--dry_run]`
    - 顺序启动所有 prefill、decode 实例；若配置中定义了 `proxy` 则再启动代理。
    - 默认轮询 `http://localhost:<端口>/health` 直至就绪（`--no_wait` 跳过）。
    - `--dry_run` 只打印完整命令和环境变量，不实际启动。
  - `python pd_service_ctl.py stop [--config <yaml>]`
    - 指定配置文件：按配置中实例名逐个停止。
    - 不指定：扫描 `/tmp/vllm_*.pid` 停止所有残留实例。
- **代码调用**：
  ```python
  from pd_service_ctl import load_config, PdServiceCtl
  cfg = load_config(Path("configs/xxx.yaml"))
  ctl = PdServiceCtl(cfg)
  ctl.start_stack(Path("logs"))
  ctl.stop()
  ```
- **说明**：`NIC_NAME` / `LOCAL_IP` 在配置中设为 `null` 时自动从 `ip` / `ifconfig` 探测。

---

## `pd_proxy.py`

内置负载均衡代理（基于 vLLM `disagg_proxy_demo.py` 精简），支持多 P/D 实例轮询调度。当配置文件中定义了 `proxy` 段时，`pd_service_ctl.py` 会自动启动该代理。也可独立使用：

```bash
python pd_proxy.py --model qwen3_32b \
    --prefill localhost:9000 \
    --decode localhost:9010 localhost:9011 \
    --port 8000
```

---

## `analysis/` 分析脚本

均在 **`analysis/`** 下执行；输入一般为 **包含若干 `batch_*` 子目录的父目录**，输出多为 **Excel（`.xlsx`）**。需安装 **`openpyxl`**。

| 脚本 | 功能概要 |
|------|----------|
| **`parse_aisbench_log.py`** | 从各 `batch_*/aisbench.log` 解析 ais_bench 的 **Performance Results** 等表格（E2EL、TTFT、TPOT、吞吐等），生成 **汇总表 + 多指标折线图**。 |
| **`parse_kv_transfer_decode_log.py`** | 从各 `batch_*` 下 **decode.log** 解析 **KV cache 传输** 日志行，统计并输出 **汇总 + 折线图**。 |
| **`vllm_engine_metrics_plot.py`** | 从 **prefill.log / decode*.log** 解析 vLLM **Engine** 周期指标（吞吐、Running/Waiting、KV 等），输出 **原始数据表 + 折线图**。 |
| **`throughput_concurrency_sweep_to_excel.py`** | 从各 `batch_*` 的 prefill/decode 日志解析 **吞吐与并发** 行，生成 **宽表** 便于多档位对比。 |

**基本用法示例**：

```bash
cd /path/to/pd_disaggregation/analysis

python parse_aisbench_log.py /path/to/run_dir
python parse_kv_transfer_decode_log.py /path/to/run_dir
python vllm_engine_metrics_plot.py /path/to/run_dir
python throughput_concurrency_sweep_to_excel.py /path/to/run_dir
```
