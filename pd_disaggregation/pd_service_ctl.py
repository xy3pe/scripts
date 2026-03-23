#!/usr/bin/env python3
"""
PD 分离：配置驱动的 vLLM Prefill / Decode / 负载均衡代理 启停控制。

两种用法：
  1) 命令行：
       python pd_service_ctl.py start --config configs/xxx.yaml [--log_dir logs/] [--dry_run]
       python pd_service_ctl.py stop  [--config configs/xxx.yaml]
  2) 代码：
       from pd_service_ctl import load_config, PdServiceCtl
       cfg = load_config(Path("configs/xxx.yaml"))
       ctl = PdServiceCtl(cfg)
       ctl.start_stack(Path("logs"))
       ctl.stop()
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# 常量 & 工具
# ---------------------------------------------------------------------------

PKG_DIR = Path(__file__).resolve().parent

_FALLBACK_NIC_NAME = "eth0"
_FALLBACK_LOCAL_IP = "172.17.0.4"

LogFn = Callable[[str], None]


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_default(msg: str) -> None:
    print(f"{ts()} [pd] {msg}")


# ---------------------------------------------------------------------------
# 网卡 / IP 自动探测（保留原有逻辑）
# ---------------------------------------------------------------------------


def _ipv4_on_dev(dev: str) -> Optional[str]:
    try:
        p = subprocess.run(
            ["ip", "-json", "addr", "show", "dev", dev],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        data = json.loads(p.stdout)
        if not data:
            return None
        for info in data[0].get("addr_info", []):
            if info.get("family") != "inet":
                continue
            loc = info.get("local")
            if loc and not loc.startswith("127."):
                return loc
    except (json.JSONDecodeError, OSError, KeyError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ip_json() -> Optional[Tuple[str, str]]:
    try:
        p = subprocess.run(
            ["ip", "-json", "route", "show", "default"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        routes = json.loads(p.stdout)
        if not routes:
            return None
        r0 = routes[0]
        dev = r0.get("dev")
        if not dev:
            return None
        prefsrc = r0.get("prefsrc")
        if prefsrc:
            return dev, prefsrc
        ip = _ipv4_on_dev(dev)
        if ip:
            return dev, ip
    except (json.JSONDecodeError, OSError, KeyError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ip_text() -> Optional[Tuple[str, str]]:
    try:
        p = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if p.returncode != 0 or not (p.stdout or "").strip():
            return None
        line = (p.stdout or "").strip().splitlines()[0]
        m = re.search(r"\bdefault\s+(?:via\s+\S+\s+)?dev\s+(\S+)", line)
        if not m:
            return None
        dev = m.group(1)
        m2 = re.search(r"\bsrc\s+(\d+\.\d+\.\d+\.\d+)", line)
        if m2:
            return dev, m2.group(1)
        ip = _ipv4_on_dev(dev)
        if ip:
            return dev, ip
    except (OSError, IndexError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nic_ip_via_ifconfig() -> Optional[Tuple[str, str]]:
    for cmd in (["ifconfig"], ["ifconfig", "-a"]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=8, check=False)
            if p.returncode != 0:
                continue
            text = p.stdout or ""
        except OSError:
            continue
        for block in text.split("\n\n"):
            lines = [ln for ln in block.strip().splitlines() if ln.strip()]
            if not lines:
                continue
            m0 = re.match(r"^(\S+):", lines[0].strip())
            if not m0:
                continue
            iface = m0.group(1)
            if iface == "lo":
                continue
            rest = "\n".join(lines[1:])
            m = re.search(r"inet (?:addr:)?(\d+\.\d+\.\d+\.\d+)", rest)
            if not m:
                continue
            ip = m.group(1)
            if ip.startswith("127."):
                continue
            return iface, ip
    return None


def _detect_default_nic_ip() -> Tuple[str, str]:
    for fn in (_detect_nic_ip_via_ip_json, _detect_nic_ip_via_ip_text, _detect_nic_ip_via_ifconfig):
        t = fn()
        if t:
            return t
    print(
        f"{ts()} [pd] NIC_NAME/LOCAL_IP 探测失败，"
        f"使用回退值: NIC_NAME={_FALLBACK_NIC_NAME!r} LOCAL_IP={_FALLBACK_LOCAL_IP!r}",
        file=sys.stderr,
    )
    return _FALLBACK_NIC_NAME, _FALLBACK_LOCAL_IP


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------


@dataclass
class InstanceConfig:
    """一个 vLLM 引擎实例（prefill 或 decode）。"""
    name: str                       # "P0", "D0", "D1"
    role: str                       # "prefill" | "decode"
    port: int
    devices: str                    # ASCEND_RT_VISIBLE_DEVICES, 如 "0,1"
    tensor_parallel_size: int
    dp_size: int                    # 同角色实例总数（自动计算）
    dp_rank: int                    # 在同角色列表中的 index（自动计算）
    dp_port: int
    kv_port: int
    engine_id: int
    hccl_buffsize: int
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterConfig:
    """从 YAML 解析出的完整集群配置。"""
    cluster_name: str
    model_path: str
    served_model_name: str
    vllm_venv: Path
    # 日志
    log_dir: Path
    # 路径
    transfer_engine_lib: str
    python_lib: str
    code_root: Path
    proxy_script_rel: str
    # 网络
    nic_name: str
    local_ip: str
    # vLLM 默认参数 & KV connector
    vllm_defaults: Dict[str, Any]
    kv_connector: Dict[str, str]
    # 实例
    prefill_instances: List[InstanceConfig]
    decode_instances: List[InstanceConfig]
    # 代理
    proxy_port: Optional[int]       # None = 不启代理
    # 就绪等待
    ready_timeout_s: int = 300
    proxy_sleep_s: int = 10


# ---------------------------------------------------------------------------
# YAML 加载 & 校验
# ---------------------------------------------------------------------------


def load_config(path: Path) -> ClusterConfig:
    """读取 YAML 配置文件并返回 ``ClusterConfig``。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 网络：null → 自动探测
    net = raw.get("network") or {}
    nic_name = net.get("nic_name")
    local_ip = net.get("local_ip")
    if not nic_name or not local_ip:
        detected_nic, detected_ip = _detect_default_nic_ip()
        nic_name = nic_name or detected_nic
        local_ip = local_ip or detected_ip

    # 解析实例
    prefill_raw = raw.get("prefill") or []
    decode_raw = raw.get("decode") or []
    prefill_count = len(prefill_raw)
    decode_count = len(decode_raw)

    def _parse_instances(items: list, role: str, count: int) -> List[InstanceConfig]:
        result: List[InstanceConfig] = []
        for idx, item in enumerate(items):
            result.append(InstanceConfig(
                name=item["name"],
                role=role,
                port=item["port"],
                devices=str(item["devices"]),
                tensor_parallel_size=item["tensor_parallel_size"],
                dp_size=count,
                dp_rank=idx,
                dp_port=item["dp_port"],
                kv_port=item["kv_port"],
                engine_id=item["engine_id"],
                hccl_buffsize=item.get("hccl_buffsize", 256 if role == "prefill" else 1024),
                overrides=item.get("overrides") or {},
            ))
        return result

    prefill_instances = _parse_instances(prefill_raw, "prefill", prefill_count)
    decode_instances = _parse_instances(decode_raw, "decode", decode_count)

    # 代理
    proxy_raw = raw.get("proxy")
    proxy_port = proxy_raw["port"] if proxy_raw else None

    # venv
    venv = raw.get("venv") or {}
    paths = raw.get("paths") or {}

    cfg = ClusterConfig(
        cluster_name=raw.get("cluster_name", "unnamed"),
        model_path=raw["model"]["path"],
        served_model_name=raw["model"].get("served_name", "default"),
        vllm_venv=Path(venv["vllm"]),
        log_dir=Path(raw.get("log_dir", "logs")),
        transfer_engine_lib=paths.get("transfer_engine_lib", "/usr/local/lib"),
        python_lib=paths.get("python_lib", ""),
        code_root=Path(paths.get("code_root", "/root/autodl-tmp/code")),
        proxy_script_rel=paths.get("proxy_script", ""),
        nic_name=nic_name,
        local_ip=local_ip,
        vllm_defaults=raw.get("vllm_defaults") or {},
        kv_connector=raw.get("kv_connector") or {},
        prefill_instances=prefill_instances,
        decode_instances=decode_instances,
        proxy_port=proxy_port,
    )

    # 校验
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: ClusterConfig) -> None:
    """基本校验：端口唯一、设备不重叠。"""
    all_inst = cfg.prefill_instances + cfg.decode_instances
    ports = [inst.port for inst in all_inst]
    if len(ports) != len(set(ports)):
        raise ValueError(f"实例端口有重复: {ports}")
    devices_sets = [(inst.name, set(inst.devices.split(","))) for inst in all_inst]
    for i in range(len(devices_sets)):
        for j in range(i + 1, len(devices_sets)):
            overlap = devices_sets[i][1] & devices_sets[j][1]
            if overlap:
                raise ValueError(
                    f"实例 {devices_sets[i][0]} 和 {devices_sets[j][0]} 设备重叠: {overlap}"
                )


# ---------------------------------------------------------------------------
# PID 文件管理
# ---------------------------------------------------------------------------

def _pid_file(name: str) -> Path:
    """返回实例对应的 PID 文件路径。"""
    return Path(f"/tmp/vllm_{name}.pid")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _direct_child_pids(pid: int) -> List[int]:
    try:
        out = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return []
    kids: List[int] = []
    for line in (out.stdout or "").strip().splitlines():
        line = line.strip()
        if line:
            try:
                kids.append(int(line))
            except ValueError:
                continue
    return kids


def _collect_pid_tree(root: int) -> set:
    seen: set = set()
    stack = [root]
    while stack:
        p = stack.pop()
        if p <= 0 or p in seen:
            continue
        seen.add(p)
        for c in _direct_child_pids(p):
            if c not in seen:
                stack.append(c)
    return seen


def _kill_pid_tree(pid: int, log: LogFn, label: str, wait_cap: int = 30) -> None:
    if not _pid_alive(pid):
        return
    tree = _collect_pid_tree(pid)
    others = sorted((tree - {pid}), reverse=True)
    for p in others:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    t0 = time.time()
    while time.time() - t0 < wait_cap:
        if not _pid_alive(pid):
            return
        time.sleep(0.2)

    if _pid_alive(pid):
        log(f"[stop {label}] pid {pid} 在 {wait_cap}s 内仍存在，放弃。")


def _stop_by_pid_file(pid_file: Path, label: str, log: LogFn) -> None:
    """读取 PID 文件并停止对应进程树。"""
    if not pid_file.is_file():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except ValueError:
        pid_file.unlink(missing_ok=True)
        return
    if not _pid_alive(pid):
        pid_file.unlink(missing_ok=True)
        return
    log(f"[stop {label}] 结束 PID {pid}（含子进程）...")
    _kill_pid_tree(pid, log, label)
    pid_file.unlink(missing_ok=True)
    log(f"[stop {label}] 已停止。")


# ---------------------------------------------------------------------------
# LD_PRELOAD 探测
# ---------------------------------------------------------------------------

_LD_PRELOAD_CACHE: Optional[str] = None


def _resolve_ld_preload() -> str:
    global _LD_PRELOAD_CACHE
    if _LD_PRELOAD_CACHE is not None:
        return _LD_PRELOAD_CACHE

    stdcxx = ""
    for candidate in (
        "/usr/lib/aarch64-linux-gnu/libstdc++.so.6",
        "/usr/lib64/libstdc++.so.6",
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
    ):
        if Path(candidate).is_file():
            stdcxx = candidate
            break

    jemalloc = "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
    parts = [p for p in [jemalloc, stdcxx] if p]
    _LD_PRELOAD_CACHE = ":".join(parts)
    return _LD_PRELOAD_CACHE


def _resolve_ld_library_path(cfg: ClusterConfig) -> str:
    parts = []
    if cfg.python_lib:
        parts.append(cfg.python_lib)
    if cfg.transfer_engine_lib:
        parts.append(cfg.transfer_engine_lib)
    parts.extend(["/usr/lib64", "/usr/lib/aarch64-linux-gnu", "/usr/lib"])
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    return ":".join(parts)


# ---------------------------------------------------------------------------
# 命令构建
# ---------------------------------------------------------------------------


def _build_env(cfg: ClusterConfig, inst: InstanceConfig) -> Dict[str, str]:
    """构建一个 vLLM 实例的完整环境变量字典。"""
    omp_threads = str(cfg.vllm_defaults.get("omp_num_threads", 10))

    env = os.environ.copy()
    env.update({
        "ASCEND_RT_VISIBLE_DEVICES": inst.devices,
        "LD_PRELOAD": _resolve_ld_preload(),
        "LD_LIBRARY_PATH": _resolve_ld_library_path(cfg),
        "HCCL_IF_IP": cfg.local_ip,
        "GLOO_SOCKET_IFNAME": cfg.nic_name,
        "TP_SOCKET_IFNAME": cfg.nic_name,
        "HCCL_SOCKET_IFNAME": cfg.nic_name,
        "OMP_PROC_BIND": "false",
        "OMP_NUM_THREADS": omp_threads,
        "HCCL_BUFFSIZE": str(inst.hccl_buffsize),
        "VLLM_DP_SIZE": str(inst.dp_size),
        "VLLM_DP_MASTER_IP": "127.0.0.1",
        "VLLM_DP_MASTER_PORT": str(inst.dp_port),
        "VLLM_DP_RANK_LOCAL": "0",
        "VLLM_DP_RANK": str(inst.dp_rank),
        "VLLM_DP_SIZE_LOCAL": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "TASK_QUEUE_ENABLE": "1",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork",
        "VLLM_ASCEND_EXTERNAL_DP_LB_ENABLED": "1",
    })
    return env


def _build_kv_transfer_config(cfg: ClusterConfig, inst: InstanceConfig) -> str:
    """构建 --kv-transfer-config JSON 字符串。"""
    kv_role = "kv_producer" if inst.role == "prefill" else "kv_consumer"

    # 自动推算 dp_size：取同角色实例列表长度
    prefill_dp = len(cfg.prefill_instances)
    prefill_tp = cfg.prefill_instances[0].tensor_parallel_size if cfg.prefill_instances else 2
    decode_dp = len(cfg.decode_instances)
    decode_tp = cfg.decode_instances[0].tensor_parallel_size if cfg.decode_instances else 2

    kv_config = {
        "kv_connector": cfg.kv_connector.get("type", "MooncakeConnectorV1"),
        "kv_buffer_device": cfg.kv_connector.get("buffer_device", "npu"),
        "kv_role": kv_role,
        "kv_parallel_size": "1",
        "kv_port": str(inst.kv_port),
        "engine_id": str(inst.engine_id),
        "kv_connector_extra_config": {
            "prefill": {"dp_size": prefill_dp, "tp_size": prefill_tp},
            "decode": {"dp_size": decode_dp, "tp_size": decode_tp},
        },
        "kv_connector_module_path": cfg.kv_connector.get(
            "module_path", "vllm_ascend.distributed.mooncake_connector"
        ),
    }
    return json.dumps(kv_config)


def _build_vllm_args(cfg: ClusterConfig, inst: InstanceConfig) -> List[str]:
    """构建 ``vllm serve ...`` 的完整参数列表。"""
    defaults = cfg.vllm_defaults
    merged = {**defaults, **inst.overrides}

    args = [
        "vllm", "serve", cfg.model_path,
        "--host", "0.0.0.0",
        "--port", str(inst.port),
        "--tensor-parallel-size", str(inst.tensor_parallel_size),
        "--served-model-name", cfg.served_model_name,
    ]

    # dtype
    if merged.get("dtype"):
        args.extend(["--dtype", str(merged["dtype"])])

    # 数值参数
    for key, flag in [
        ("max_model_len", "--max-model-len"),
        ("max_num_batched_tokens", "--max-num-batched-tokens"),
        ("max_num_seqs", "--max-num-seqs"),
        ("gpu_memory_utilization", "--gpu-memory-utilization"),
        ("seed", "--seed"),
    ]:
        if key in merged:
            args.extend([flag, str(merged[key])])

    # 布尔开关
    for key, flag in [
        ("enforce_eager", "--enforce-eager"),
        ("trust_remote_code", "--trust-remote-code"),
        ("enable_auto_tool_choice", "--enable-auto-tool-choice"),
    ]:
        if merged.get(key):
            args.append(flag)

    # tool-call-parser
    if merged.get("tool_call_parser"):
        args.extend(["--tool-call-parser", str(merged["tool_call_parser"])])

    # decode 多实例时加 data-parallel 参数
    if inst.role == "decode" and inst.dp_size > 1:
        args.extend([
            "--nnodes", "1",
            "--data-parallel-size", str(inst.dp_size),
            "--data-parallel-rank", str(inst.dp_rank),
            "--data-parallel-address", "127.0.0.1",
            "--data-parallel-rpc-port", str(inst.dp_port),
            "--data-parallel-size-local", "1",
        ])

    # kv-transfer-config
    args.extend(["--kv-transfer-config", _build_kv_transfer_config(cfg, inst)])

    return args


def _build_proxy_args(cfg: ClusterConfig) -> List[str]:
    """构建代理启动参数列表。"""
    proxy_script = cfg.code_root / cfg.proxy_script_rel
    venv_python = cfg.vllm_venv / "bin" / "python3"

    argv = [
        str(venv_python), str(proxy_script),
        "--port", str(cfg.proxy_port),
        "--host", "0.0.0.0",
    ]

    # prefill hosts/ports
    argv.append("--prefiller-hosts")
    for _ in cfg.prefill_instances:
        argv.append(cfg.local_ip)
    argv.append("--prefiller-ports")
    for inst in cfg.prefill_instances:
        argv.append(str(inst.port))

    # decode hosts/ports
    argv.append("--decoder-hosts")
    for _ in cfg.decode_instances:
        argv.append(cfg.local_ip)
    argv.append("--decoder-ports")
    for inst in cfg.decode_instances:
        argv.append(str(inst.port))

    return argv


# ---------------------------------------------------------------------------
# Dry-run 输出
# ---------------------------------------------------------------------------


def dry_run(cfg: ClusterConfig, log_dir: Path) -> None:
    """打印所有实例的启动命令和环境变量（不实际执行）。"""
    log_dir = log_dir.resolve()
    all_inst = cfg.prefill_instances + cfg.decode_instances

    print(f"{'=' * 72}")
    print(f"  Cluster: {cfg.cluster_name}")
    print(f"  Model:   {cfg.model_path}")
    print(f"  Network: NIC={cfg.nic_name} IP={cfg.local_ip}")
    print(f"  Log dir: {log_dir}")
    print(f"{'=' * 72}\n")

    for inst in all_inst:
        env = _build_env(cfg, inst)
        args = _build_vllm_args(cfg, inst)

        # 只打印与父进程不同的环境变量
        diff_env = {k: v for k, v in env.items() if os.environ.get(k) != v}

        role_tag = inst.role.upper()
        log_file = _instance_log_file(inst, log_dir)

        print(f"--- [{role_tag}] {inst.name} (port={inst.port}, devices={inst.devices}) ---")
        print(f"  PID file: {_pid_file(inst.name)}")
        print(f"  Log file: {log_file}")
        print(f"  Key env vars:")
        for k in sorted(diff_env):
            if k in ("PATH", "HOME", "USER", "SHELL", "TERM", "LANG", "LC_ALL",
                      "LOGNAME", "HOSTNAME", "SHLVL", "_"):
                continue
            print(f"    {k}={diff_env[k]}")
        print(f"  Command:")
        print(f"    {' '.join(args)}")
        print()

    if cfg.proxy_port is not None:
        proxy_args = _build_proxy_args(cfg)
        print(f"--- [PROXY] (port={cfg.proxy_port}) ---")
        print(f"  PID file: {_pid_file('proxy')}")
        print(f"  Log file: {log_dir / 'proxy.log'}")
        print(f"  Command:")
        print(f"    {' '.join(proxy_args)}")
        print()


# ---------------------------------------------------------------------------
# 服务编排
# ---------------------------------------------------------------------------


def _instance_log_file(inst: InstanceConfig, log_dir: Path) -> Path:
    """返回实例日志文件路径。"""
    return log_dir / f"{inst.role}_{inst.name}.log"


class PdServiceCtl:
    """
    PD 服务编排：基于 ClusterConfig 启动 / 停止所有 vLLM 实例和代理。

    ``log`` 为行级回调；外部调用者可传入自定义日志函数。
    """

    def __init__(self, cfg: ClusterConfig, log: LogFn = log_default) -> None:
        self._cfg = cfg
        self._log = log

    @property
    def cfg(self) -> ClusterConfig:
        return self._cfg

    @property
    def has_proxy(self) -> bool:
        return self._cfg.proxy_port is not None

    # ---------- 启动单个实例 ----------

    def start_instance(self, inst: InstanceConfig, log_dir: Path) -> None:
        """启动一个 vLLM 实例（prefill 或 decode）。"""
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        env = _build_env(self._cfg, inst)
        args = _build_vllm_args(self._cfg, inst)
        log_file = _instance_log_file(inst, log_dir)

        venv_activate = self._cfg.vllm_venv / "bin" / "activate"
        if not venv_activate.is_file():
            raise FileNotFoundError(f"venv activate 不存在: {venv_activate}")

        # 通过 bash source activate 然后执行 vllm serve
        cmd_str = " ".join(args)
        inner = f"""
set -euo pipefail
source "{venv_activate}"
set -m
nohup {cmd_str} >> "{log_file}" 2>&1 &
echo $! > "{_pid_file(inst.name)}"
"""
        self._log(f"启动 {inst.role} [{inst.name}] (port={inst.port}, devices={inst.devices})...")
        subprocess.run(
            ["/bin/bash", "-c", inner],
            check=True,
            cwd=str(log_dir),
            env=env,
        )

    # ---------- 启动代理 ----------

    def start_proxy(self, log_dir: Path) -> None:
        """启动负载均衡代理。"""
        if self._cfg.proxy_port is None:
            self._log("配置中未定义 proxy，跳过。")
            return

        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        proxy_script = self._cfg.code_root / self._cfg.proxy_script_rel
        if not proxy_script.is_file():
            raise FileNotFoundError(f"代理脚本不存在: {proxy_script}")

        proxy_args = _build_proxy_args(self._cfg)
        cmd_str = " ".join(f'"{a}"' if " " in a else a for a in proxy_args)
        log_file = log_dir / "proxy.log"

        inner = f"""
set -euo pipefail
nohup {cmd_str} >> "{log_file}" 2>&1 &
echo $! > "{_pid_file('proxy')}"
"""
        env = os.environ.copy()
        env["LOCAL_IP"] = self._cfg.local_ip

        self._log(f"启动代理 (port={self._cfg.proxy_port})...")
        subprocess.run(
            ["/bin/bash", "-c", inner],
            check=True,
            cwd=str(log_dir),
            env=env,
        )

    # ---------- 端口就绪等待 ----------

    def wait_for_port(
        self,
        port: int,
        name: str,
        timeout_s: Optional[int] = None,
    ) -> bool:
        t = self._cfg.ready_timeout_s if timeout_s is None else timeout_s
        self._log(f"等待 {name} 端口 {port} 就绪（超时 {t}s）...")
        elapsed = 0
        url = f"http://localhost:{port}/health"
        step = 5
        while elapsed < t:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        self._log(f"{name} 已就绪（已等待 {elapsed}s）。")
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(step)
            elapsed += step
        self._log(f"ERROR: {name} 在 {t}s 内未就绪。")
        return False

    # ---------- 一键启动 ----------

    def start_stack(
        self,
        log_dir: Path,
        *,
        with_proxy: Optional[bool] = None,
        wait_ready: bool = True,
    ) -> int:
        """
        顺序启动 prefill → decode →（可选）代理。

        ``with_proxy`` 为 ``None`` 时：配置中有 proxy 则启。
        返回 0 成功。
        """
        log_dir = log_dir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)

        wp = self.has_proxy if with_proxy is None else with_proxy
        if wp and not self.has_proxy:
            self._log("ERROR: 指定了代理但配置中未定义 proxy")
            return 1

        # 启动 prefill
        for inst in self._cfg.prefill_instances:
            try:
                self.start_instance(inst, log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: {inst.name} 启动失败: {e}")
                self.stop()
                return 1

        if wait_ready:
            for inst in self._cfg.prefill_instances:
                if not self.wait_for_port(inst.port, f"prefill[{inst.name}]"):
                    self.stop()
                    return 1

        # 启动 decode
        for inst in self._cfg.decode_instances:
            try:
                self.start_instance(inst, log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: {inst.name} 启动失败: {e}")
                self.stop()
                return 1

        if wait_ready:
            for inst in self._cfg.decode_instances:
                if not self.wait_for_port(inst.port, f"decode[{inst.name}]"):
                    self.stop()
                    return 1

        # 启动代理
        if wp:
            try:
                self.start_proxy(log_dir)
            except (FileNotFoundError, subprocess.CalledProcessError) as e:
                self._log(f"ERROR: 代理启动失败: {e}")
                self.stop()
                return 1
            time.sleep(self._cfg.proxy_sleep_s)

        self._log(f"PD 服务已拉起，日志目录: {log_dir}")
        return 0

    # ---------- 停止 ----------

    def stop(self) -> None:
        """停止所有实例：代理 → decode → prefill。"""
        # 代理
        _stop_by_pid_file(_pid_file("proxy"), "proxy", self._log)

        # decode（逆序）
        for inst in reversed(self._cfg.decode_instances):
            _stop_by_pid_file(_pid_file(inst.name), inst.name, self._log)

        # prefill（逆序）
        for inst in reversed(self._cfg.prefill_instances):
            _stop_by_pid_file(_pid_file(inst.name), inst.name, self._log)

    @staticmethod
    def stop_all(log: LogFn = log_default) -> None:
        """扫描 /tmp/vllm_*.pid 停止所有残留实例（无需配置文件）。"""
        pid_files = sorted(glob.glob("/tmp/vllm_*.pid"))
        if not pid_files:
            log("未找到 /tmp/vllm_*.pid，无需停止。")
            return
        for pf_str in pid_files:
            pf = Path(pf_str)
            label = pf.stem.replace("vllm_", "")
            _stop_by_pid_file(pf, label, log)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PD Prefill / Decode / 代理 启停（配置驱动）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="启动 prefill → decode →（可选）代理")
    p_start.add_argument(
        "--config", "-c",
        type=Path,
        required=True,
        help="YAML 配置文件路径",
    )
    p_start.add_argument(
        "--log_dir",
        type=Path,
        default=None,
        help="日志目录（默认使用配置文件中的 log_dir）",
    )
    p_start.add_argument("--no_wait", action="store_true", help="不等待 /health")
    p_start.add_argument("--dry_run", action="store_true", help="只打印命令，不实际启动")

    p_stop = sub.add_parser("stop", help="停止所有实例")
    p_stop.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="YAML 配置文件（不指定则扫描 /tmp/vllm_*.pid）",
    )

    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_cli_parser().parse_args(argv)

    if args.cmd == "stop":
        if args.config:
            cfg = load_config(args.config)
            PdServiceCtl(cfg).stop()
        else:
            PdServiceCtl.stop_all()
        return 0

    # start
    cfg = load_config(args.config)

    log_dir = (args.log_dir if args.log_dir is not None else cfg.log_dir).resolve()

    if args.dry_run:
        dry_run(cfg, log_dir)
        return 0

    ctl = PdServiceCtl(cfg)
    return ctl.start_stack(log_dir, with_proxy=None, wait_ready=not args.no_wait)


if __name__ == "__main__":
    raise SystemExit(main())
