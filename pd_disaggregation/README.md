# PD 分离（Prefill / Decode）编排与结果分析

本目录包含 **PD 分离推理** 的启停编排、ShareGPT 批量压测，以及压测日志的离线分析脚本。拓扑与脚本以 **`pd_service_ctl.py` 所在目录** 为根：`--pd_mode` 指向其下的子目录（如 `Qwen3-32B/1P2_2D2`），内含 `run_prefill.sh`、`run_decode.sh`，若存在 `pd_proxy.py` 则可选负载均衡代理。

---

## `pd_service_ctl.py`

**作用**：统一拉起 / 停止 **Prefill → Decode →（可选）代理**。

- **命令行**
  - `python pd_service_ctl.py start [--pd_mode …] [--log_dir …] [--nic_name …] [--local_ip …] [--no_wait]`
    - 顺序启动 prefill、decode；若当前拓扑目录存在 `pd_proxy.py` 则再启动代理；默认轮询 `http://localhost:<端口>/health` 直至就绪（可用 `--no_wait` 跳过）。
  - `python pd_service_ctl.py stop`
    - 停止代理（若曾启动）、decode、prefill（按进程树回收，PID 见 `/tmp/vllm_*.pid`）。
- **代码调用**：构造 `PdRuntimeConfig` + `PdServiceCtl`，使用 `start_prefill` / `start_decode` / `start_proxy`、`start_stack`、`stop` 等（详见 `--help` 与源码）。
- **说明**：`NIC_NAME` / `LOCAL_IP` 默认在导入时尝试从 `ip` / `ifconfig` 探测，失败则回退到模块内常量（失败时会在 stderr 打印提示）。

---

## `run_sharegpt_sweep.py`

**作用**：按 **batch 列表** 循环执行：起 PD 栈 → 跑 **ais_bench**（ShareGPT）→ 停栈；日志与 `pd_service_ctl` 对齐。

- **典型流程（每轮）**：`PdServiceCtl.start_stack`（prefill → decode → 代理）→ 修改 benchmark 配置中的 `batch_size` → 调用 `ais_bench` → `stop`。
- **默认输出目录**：`<本脚本目录>/logs/sharegpt`（可用 `--run_dir` 覆盖），其下含 `sweep.log`、`batch_<N>/`（PD 与 aisbench 日志等）。
- **常用参数**（其余见 `--help`）：
  - `--run_dir`：本趟输出根目录  
  - `--pd_mode`：相对本目录的拓扑路径（默认与 `pd_service_ctl.PD_MODE` 一致）  
  - `--nic_name` / `--local_ip`：未指定则用 `pd_service_ctl` 探测或常量  
  - `--batch_sizes`：逗号分隔，如 `60,80,120,200`  
  - `--benchmark_dir`：ais_bench 所在工程根（脚本内默认路径可按环境修改）

**依赖**：需与本机路径一致地配置 **vLLM venv**、**tester venv**、**benchmark 目录**（见脚本顶部常量）。

---

## `analysis/` 分析脚本

均在 **`analysis/`** 下执行；输入一般为 **包含若干 `batch_*` 子目录的父目录**（与 `run_sharegpt_sweep` 产出结构一致），输出多为 **Excel（`.xlsx`）**。需安装 **`openpyxl`**。

| 脚本 | 功能概要 |
|------|----------|
| **`parse_aisbench_log.py`** | 从各 `batch_*/aisbench.log` 解析 ais_bench 的 **Performance Results** 等表格（E2EL、TTFT、TPOT、吞吐等），生成 **汇总表 + 多指标折线图**（多 Sheet）。 |
| **`parse_kv_transfer_decode_log.py`** | 从各 `batch_*` 下 **decode.log**（或 `decode_0.log`、`decode_1.log`…）解析 **KV cache 传输** 日志行，统计并输出 **汇总 + 折线图**（多 decode 时分文件对比）。 |
| **`vllm_engine_metrics_plot.py`** | 从 **prefill.log / decode*.log** 解析 vLLM **Engine** 周期指标（吞吐、Running/Waiting、KV 等），单目录或批量 `batch_*` 模式，输出 **原始数据表 + 折线图**；批量时父目录可生成 sweep 汇总表。 |
| **`throughput_concurrency_sweep_to_excel.py`** | 从各 `batch_*` 的 prefill/decode 日志解析 **吞吐与并发** 行，生成 **单 Sheet 宽表**（列名含 BS、prefill/decode 分块），便于多并发档位对比。 |

**基本用法示例**（在 `analysis/` 下，路径按实际调整）：

```bash
cd /path/to/pd_disaggregation/analysis

python parse_aisbench_log.py /path/to/run_dir
python parse_kv_transfer_decode_log.py /path/to/run_dir
python vllm_engine_metrics_plot.py /path/to/run_dir
python throughput_concurrency_sweep_to_excel.py /path/to/run_dir
```

各脚本支持 `--output` 指定输出文件名（行为以脚本 `--help` 与文件头说明为准）。

---

## 拓扑目录 `Qwen3-32B/`

各子目录（如 `1P2_1D2`、`1P2_2D2`）提供 **Prefill / Decode / 代理** 的 shell 与 `pd_proxy.py`，由 `pd_service_ctl` 注入 `LOG_DIR`、`NIC_NAME`、`LOCAL_IP` 等环境变量后启动。具体设备与模型参数以各目录内脚本为准。
