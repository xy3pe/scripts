#!/usr/bin/env python3
"""
PD 分离模式 ShareGPT 批量压测编排。

每轮：启动 prefill → 等待就绪 → 启动 decode → 可选负载均衡代理 → ais_bench → 回收进程。
启停由同目录 `pd_service_ctl.PdServiceCtl` 完成（亦可命令行 `python pd_service_ctl.py start|stop`）。
prefill / decode 为 `./<pd-mode>/` 下 `run_prefill.sh`、`run_decode.sh`；代理为 `pd_proxy.py`。

用法（任意 cwd）:
  python .../run_sharegpt_sweep.py [--run_dir ...] [--pd_mode ...] [--nic_name ...] [--local_ip ...] [--batch_sizes ...] [--benchmark_dir ...]
  ``--run_dir`` 为本趟输出根目录（默认 ``<脚本目录>/logs/sharegpt``），其下含 ``sweep.log``、``batch_*`` 等。
  端口、venv、超时、代理等待等见 ``SweepConfig`` 默认值（与 ``pd_service_ctl`` 对齐）。
"""
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

import pd_service_ctl as pdctl
from pd_service_ctl import PdRuntimeConfig, PdServiceCtl

PKG_DIR = Path(__file__).resolve().parent

VLLM_VENV = Path("/root/autodl-tmp/py_venv/vllm2")
TESTER_VENV = Path("/root/autodl-tmp/py_venv/tester")
BENCHMARK_DIR = Path("/root/autodl-tmp/code/benchmark")
CONFIG_REL = Path("ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat_multiturn.py")
LOG_ROOT = PKG_DIR / "logs"
RUN_DIR_DEFAULT = LOG_ROOT / "sharegpt"


@dataclass
class SweepConfig:
    """本趟压测输出根目录（含 sweep.log、各 batch_* 子目录）。"""
    run_dir: Path = RUN_DIR_DEFAULT
    # 未显式指定时与 pd_service_ctl.PD_MODE 一致；CLI 传入则覆盖。
    pd_mode: str = pdctl.PD_MODE
    batch_sizes: List[int] = field(default_factory=lambda: [60, 80, 120, 200])
    vllm_port: int = 9010
    proxy_port: int = 8000
    prefill_port: int = 9000
    ready_timeout_s: int = 300
    # 未显式指定时与 pd_service_ctl 模块常量一致；CLI 传入则覆盖。
    nic_name: str = pdctl.NIC_NAME
    local_ip: str = pdctl.LOCAL_IP
    vllm_venv: Path = VLLM_VENV
    tester_venv: Path = TESTER_VENV
    benchmark_dir: Path = BENCHMARK_DIR
    proxy_sleep_s: int = 10
    round_cooldown_s: int = 5

    @property
    def topo_dir(self) -> Path:
        """当前拓扑根目录：`./<pd-mode>/`。"""
        rel = self.pd_mode.strip("/").replace("\\", "/")
        return PKG_DIR / rel

    @property
    def use_proxy(self) -> bool:
        """是否存在 `pd_proxy.py`。"""
        return pdctl.pd_proxy_path(self.topo_dir) is not None


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_sweep(sweep_log: Path, msg: str) -> None:
    line = f"{ts()} [sweep] {msg}"
    print(line)
    sweep_log.parent.mkdir(parents=True, exist_ok=True)
    with sweep_log.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _pd_rt(cfg: SweepConfig) -> PdRuntimeConfig:
    return PdRuntimeConfig(
        topo_dir=cfg.topo_dir,
        vllm_venv=cfg.vllm_venv,
        pd_mode=cfg.pd_mode,
        prefill_port=cfg.prefill_port,
        vllm_port=cfg.vllm_port,
        proxy_sleep_s=cfg.proxy_sleep_s,
        ready_timeout_s=cfg.ready_timeout_s,
        nic_name=cfg.nic_name,
        local_ip=cfg.local_ip,
    )


def _batch_log_dir(cfg: SweepConfig, batch_size: int) -> Path:
    return cfg.run_dir / f"batch_{batch_size}"


def _sweep_log_fn(sweep_log: Path):
    return lambda msg: log_sweep(sweep_log, msg)


def patch_benchmark_batch_size(config_py: Path, batch_size: int) -> None:
    """将配置里 `batch_size=` 后的数值段替换为当前 batch（允许原值为空）。"""
    text = config_py.read_text(encoding="utf-8")
    new, n = re.subn(r"batch_size=\d*", f"batch_size={batch_size}", text)
    if n == 0:
        raise RuntimeError(f"未找到 batch_size= 可替换项: {config_py}")
    config_py.write_text(new, encoding="utf-8")


def run_sharegpt_benchmark(
    cfg: SweepConfig,
    batch_size: int,
    bench_port: int,
    sweep_log: Path,
) -> int:
    log_dir = cfg.run_dir / f"batch_{batch_size}"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_py = cfg.benchmark_dir / CONFIG_REL
    if not config_py.is_file():
        log_sweep(sweep_log, f"ERROR: benchmark 配置不存在: {config_py}")
        return 1

    patch_benchmark_batch_size(config_py, batch_size)
    work_dir = f"sharegpt_{cfg.pd_mode}"
    ais_log = log_dir / "aisbench.log"
    activate = cfg.tester_venv / "bin" / "activate"
    if not activate.is_file():
        log_sweep(sweep_log, f"ERROR: tester venv 不存在: {activate}")
        return 1

    inner = f"""
set -euo pipefail
export VLLM_PORT="{bench_port}"
source "{activate}"
cd "{cfg.benchmark_dir}"
ais_bench --models vllm_api_stream_chat_multiturn --datasets sharegpt_gen --mode perf --num-warmups 0 --work-dir "outputs/{work_dir}/" >> "{ais_log}" 2>&1
"""
    log_sweep(
        sweep_log,
        f"Running sharegpt benchmark (BATCH_SIZE={batch_size}, port={bench_port})...",
    )
    r = subprocess.run(["/bin/bash", "-c", inner], cwd=str(cfg.benchmark_dir))
    return r.returncode


def parse_batch_sizes(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [int(p) for p in parts]


def run_sweep(cfg: SweepConfig) -> int:
    sweep_log = cfg.run_dir / "sweep.log"
    # 进入主循环前固定本趟 sweep 是否启用代理
    use_proxy_sweep = cfg.use_proxy
    logf = _sweep_log_fn(sweep_log)
    pd_ctl = PdServiceCtl(_pd_rt(cfg), log=logf)

    def on_signal(signum: int, _frame) -> None:
        log_sweep(sweep_log, f"Interrupted (signal {signum}), cleaning up...")
        pd_ctl.stop(use_proxy=use_proxy_sweep)
        sys.exit(1)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    log_sweep(
        sweep_log,
        "Sweep start (PD mode): "
        f"RUN_DIR={cfg.run_dir} TOPO_DIR={cfg.topo_dir} "
        f"BATCH_SIZES={cfg.batch_sizes} USE_PROXY={use_proxy_sweep} "
        f"NIC_NAME={cfg.nic_name} LOCAL_IP={cfg.local_ip}",
    )

    for batch_size in cfg.batch_sizes:
        log_sweep(sweep_log, f"========== Round: BATCH_SIZE={batch_size} ==========")

        batch_log = _batch_log_dir(cfg, batch_size)
        os.environ["LOG_DIR"] = str(batch_log)

        log_sweep(
            sweep_log,
            f"PdServiceCtl.start_stack (log_dir={batch_log}, with_proxy={use_proxy_sweep})...",
        )
        stack_rc = pd_ctl.start_stack(
            batch_log,
            with_proxy=use_proxy_sweep,
            wait_ready=True,
        )
        if stack_rc != 0:
            log_sweep(
                sweep_log,
                f"ERROR: PD 栈启动失败，跳过 BATCH_SIZE={batch_size}（ctl 已尝试回收）。",
            )
            continue

        bench_port = cfg.proxy_port if use_proxy_sweep else cfg.vllm_port

        rc = run_sharegpt_benchmark(cfg, batch_size, bench_port, sweep_log)
        if rc == 0:
            log_sweep(sweep_log, f"Benchmark finished: BATCH_SIZE={batch_size}")
        else:
            log_sweep(
                sweep_log,
                f"WARNING: benchmark exited with error for BATCH_SIZE={batch_size}, continuing...",
            )

        pd_ctl.stop(use_proxy=use_proxy_sweep)
        log_sweep(
            sweep_log,
            f"vllm (PD) stopped. Waiting {cfg.round_cooldown_s}s before next round...",
        )
        time.sleep(cfg.round_cooldown_s)

    log_sweep(sweep_log, f"All rounds completed. Logs: {cfg.run_dir}")
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(description="PD 分离 ShareGPT 批量压测编排")
    p.add_argument(
        "--run_dir",
        type=Path,
        default=RUN_DIR_DEFAULT,
        help=f"本趟输出根目录（sweep.log、batch_* 等；默认 {RUN_DIR_DEFAULT}）",
    )
    p.add_argument(
        "--nic_name",
        default=None,
        help=f"网卡名（默认未指定时用 pd_service_ctl 的 NIC_NAME，当前为 {pdctl.NIC_NAME!r}）",
    )
    p.add_argument(
        "--local_ip",
        default=None,
        help=f"本机可达 IP（默认未指定时用 pd_service_ctl 的 LOCAL_IP，当前为 {pdctl.LOCAL_IP!r}）",
    )
    p.add_argument(
        "--pd_mode",
        default=None,
        help=f"相对本目录的拓扑子路径（默认未指定时用 pd_service_ctl 的 PD_MODE，当前为 {pdctl.PD_MODE!r}）",
    )
    p.add_argument(
        "--batch_sizes",
        default="200",
        help="逗号分隔的 batch 列表，如 60,80,120,200",
    )
    
    p.add_argument("--benchmark_dir", type=Path, default=BENCHMARK_DIR)
    args = p.parse_args(list(argv) if argv is not None else None)

    nic_name = args.nic_name if args.nic_name is not None else pdctl.NIC_NAME
    local_ip = args.local_ip if args.local_ip is not None else pdctl.LOCAL_IP
    pd_mode = args.pd_mode if args.pd_mode is not None else pdctl.PD_MODE

    cfg = SweepConfig(
        run_dir=args.run_dir.resolve(),
        pd_mode=pd_mode,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
        nic_name=nic_name,
        local_ip=local_ip,
        benchmark_dir=args.benchmark_dir.resolve(),
    )
    return run_sweep(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
