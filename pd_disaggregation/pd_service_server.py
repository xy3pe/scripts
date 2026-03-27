#!/usr/bin/env python3
"""
pd_service_server.py — PD 服务 HTTP 控制服务器

通过 REST API 远程控制 pd_service_ctl.py 的启动、停止、重启操作，
并提供服务状态和 NPU 显存查询。

用法:
    python pd_service_server.py --config configs/xxx.yaml [--host 0.0.0.0] [--port 8088]
                                [--log_dir logs/]

API:
    POST /start              启动全部服务
    POST /stop               停止全部服务
    POST /restart            重启全部服务（stop → 等 NPU 显存释放 → start）
    GET  /status             查询实例运行状态 + NPU HBM 用量
    GET  /task               查询最新任务进度和日志
    GET  /task/<id>          查询指定任务进度和日志

所有 POST 均返回 202 Accepted + {"task_id": "...", "op": "...", "state": "running"}。
长时任务在后台线程执行；若当前已有任务运行，返回 409 Conflict。
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from pd_service_ctl import (
    ClusterConfig,
    PdServiceCtl,
    _get_npu_hbm_usage,
    _pid_alive,
    _pid_file,
    load_config,
    log_default,
)

# ---------------------------------------------------------------------------
# 任务模型
# ---------------------------------------------------------------------------

_MAX_LOG_LINES = 2000
_TASK_HISTORY_SIZE = 10


class Task:
    """单次操作的执行状态和日志容器。"""

    def __init__(self, task_id: str, op: str) -> None:
        self.task_id = task_id
        self.op = op
        self.state = "running"          # running | done | failed
        self.rc: Optional[int] = None
        self.logs: Deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        self.logs.append(f"{ts} {msg}")
        log_default(msg)

    def finish(self, rc: int) -> None:
        self.rc = rc
        self.state = "done" if rc == 0 else "failed"
        self.end_time = time.time()

    def to_dict(self, *, tail: Optional[int] = None) -> Dict[str, Any]:
        """序列化为字典。``tail`` 为 None 时返回全量日志，否则返回最后 N 行。"""
        logs = list(self.logs)
        if tail is not None:
            logs = logs[-tail:]
        return {
            "task_id": self.task_id,
            "op": self.op,
            "state": self.state,
            "rc": self.rc,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3) if self.end_time else None,
            "elapsed_s": round((self.end_time or time.time()) - self.start_time, 1),
            "logs": logs,
        }


# ---------------------------------------------------------------------------
# 服务全局状态
# ---------------------------------------------------------------------------


class ServiceState:
    """
    持有集群配置、后台任务状态、历史记录。

    同一时间只允许一个操作运行（start/stop/restart），通过 ``_op_lock`` 串行化。
    """

    def __init__(self, cfg: ClusterConfig, log_dir: Path) -> None:
        self.cfg = cfg
        self.log_dir = log_dir
        self._op_lock = threading.Lock()
        self._current_task: Optional[Task] = None
        self._history: Deque[Task] = deque(maxlen=_TASK_HISTORY_SIZE)

    # ---------- 查询 ----------

    @property
    def current_task(self) -> Optional[Task]:
        return self._current_task

    def is_busy(self) -> bool:
        t = self._current_task
        return t is not None and t.state == "running"

    def get_task(self, task_id: Optional[str] = None) -> Optional[Task]:
        """返回指定 ID 的任务；task_id 为 None 时返回最新任务（运行中优先）。"""
        if task_id is None:
            if self._current_task:
                return self._current_task
            return self._history[0] if self._history else None
        if self._current_task and self._current_task.task_id == task_id:
            return self._current_task
        for t in self._history:
            if t.task_id == task_id:
                return t
        return None

    # ---------- 提交 ----------

    def submit(self, op: str, fn) -> Optional[Task]:
        """
        提交后台任务。fn 签名：``fn(task: Task) -> int``。

        若当前已有任务运行，返回 None（拒绝）；否则返回新建的 Task。
        """
        with self._op_lock:
            if self.is_busy():
                return None
            task = Task(uuid.uuid4().hex[:8], op)
            self._current_task = task

        thread = threading.Thread(target=self._run, args=(task, fn), daemon=True)
        thread.start()
        return task

    def _run(self, task: Task, fn) -> None:
        try:
            rc = fn(task)
            task.finish(rc if rc is not None else 0)
        except Exception as exc:
            task.log(f"ERROR: 操作异常: {exc}")
            task.finish(1)
        finally:
            self._history.appendleft(task)
            with self._op_lock:
                if self._current_task is task:
                    self._current_task = None

    # ---------- 状态查询 ----------

    def instance_status(self) -> List[Dict[str, Any]]:
        """返回各实例的存活状态（基于 PID 文件）。"""
        result: List[Dict[str, Any]] = []
        for inst in self.cfg.prefill_instances + self.cfg.decode_instances:
            pid, alive = _read_pid_alive(_pid_file(inst.name))
            result.append({
                "name": inst.name,
                "role": inst.role,
                "port": inst.port,
                "devices": inst.devices,
                "pid": pid,
                "alive": alive,
            })
        if self.cfg.proxy_port is not None:
            pid, alive = _read_pid_alive(_pid_file("proxy"))
            result.append({
                "name": "proxy",
                "role": "proxy",
                "port": self.cfg.proxy_port,
                "devices": None,
                "pid": pid,
                "alive": alive,
            })
        return result

    def npu_hbm_status(self) -> Optional[Dict[str, int]]:
        """返回 NPU HBM 用量（MB）字典，key 为 'npu{index}'。"""
        hbm = _get_npu_hbm_usage(lambda _: None)
        if hbm is None:
            return None
        return {f"npu{k}": v for k, v in sorted(hbm.items())}


def _read_pid_alive(pid_file: Path):
    """读取 PID 文件并判断进程是否存活，返回 (pid, alive)。"""
    if not pid_file.is_file():
        return None, False
    try:
        pid = int(pid_file.read_text().strip())
        return pid, _pid_alive(pid)
    except (ValueError, OSError):
        return None, False


# ---------------------------------------------------------------------------
# HTTP 请求处理
# ---------------------------------------------------------------------------

_state: Optional[ServiceState] = None


def _json_response(handler: BaseHTTPRequestHandler, code: int, body: Any) -> None:
    data = json.dumps(body, ensure_ascii=False, indent=2).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json_body(handler: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length", 0))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return json.loads(raw) or {}
    except json.JSONDecodeError:
        return {}


class PdControlHandler(BaseHTTPRequestHandler):
    """HTTP 请求路由。"""

    def log_message(self, fmt, *args):
        log_default(f"HTTP [{self.address_string()}] {fmt % args}")

    # ---------- GET ----------

    def do_GET(self):
        path = self.path.rstrip("/")
        if path in ("", "/status"):
            self._status()
        elif path == "/task" or path.startswith("/task/"):
            self._get_task(path)
        else:
            _json_response(self, 404, {"error": f"unknown path: {self.path}"})

    def _status(self):
        instances = _state.instance_status()
        npu = _state.npu_hbm_status()
        alive_count = sum(1 for i in instances if i["alive"])
        task = _state.current_task
        _json_response(self, 200, {
            "busy": _state.is_busy(),
            "alive_instances": alive_count,
            "total_instances": len(instances),
            "current_task": task.to_dict(tail=20) if task else None,
            "instances": instances,
            "npu_hbm_mb": npu,
        })

    def _get_task(self, path: str):
        parts = [p for p in path.split("/") if p]
        task_id = parts[1] if len(parts) >= 2 else None
        task = _state.get_task(task_id)
        if task is None:
            _json_response(self, 404, {"error": "task not found"})
            return
        _json_response(self, 200, task.to_dict())

    # ---------- POST ----------

    def do_POST(self):
        path = self.path.rstrip("/")
        if path == "/start":
            self._start()
        elif path == "/stop":
            self._stop()
        elif path == "/restart":
            self._restart()
        else:
            _json_response(self, 404, {"error": f"unknown path: {self.path}"})

    def _start(self):
        body = _read_json_body(self)
        no_wait = bool(body.get("no_wait", False))
        log_dir = Path(body["log_dir"]) if "log_dir" in body else _state.log_dir

        def fn(task: Task) -> int:
            return PdServiceCtl(_state.cfg, log=task.log).start_stack(
                log_dir, wait_ready=not no_wait
            )

        self._submit("start", fn)

    def _stop(self):
        def fn(task: Task) -> int:
            PdServiceCtl(_state.cfg, log=task.log).stop()
            return 0

        self._submit("stop", fn)

    def _restart(self):
        body = _read_json_body(self)
        no_wait = bool(body.get("no_wait", False))
        mem_threshold_mb = int(body.get("mem_threshold_mb", 5000))
        mem_timeout_s = int(body.get("mem_timeout_s", 300))
        log_dir = Path(body["log_dir"]) if "log_dir" in body else _state.log_dir

        def fn(task: Task) -> int:
            return PdServiceCtl(_state.cfg, log=task.log).restart(
                log_dir,
                mem_threshold_mb=mem_threshold_mb,
                mem_timeout_s=mem_timeout_s,
                wait_ready=not no_wait,
            )

        self._submit("restart", fn)

    def _submit(self, op: str, fn) -> None:
        task = _state.submit(op, fn)
        if task is None:
            current = _state.current_task
            _json_response(self, 409, {
                "error": "busy — another operation is already running",
                "current_task": current.to_dict(tail=5) if current else None,
            })
            return
        _json_response(self, 202, {
            "task_id": task.task_id,
            "op": op,
            "state": "running",
        })


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    global _state

    parser = argparse.ArgumentParser(
        description="PD 服务 HTTP 控制服务器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", type=Path, required=True, help="YAML 配置文件路径")
    parser.add_argument("--log_dir", type=Path, default=None, help="日志目录（默认使用配置中的 log_dir）")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8088, help="监听端口")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log_dir = (args.log_dir if args.log_dir is not None else cfg.log_dir).resolve()
    _state = ServiceState(cfg, log_dir)

    log_default(f"PD 控制服务器启动  http://{args.host}:{args.port}")
    log_default(f"配置文件: {args.config.resolve()}")
    log_default(f"日志目录: {log_dir}")
    log_default("接口: POST /start  POST /stop  POST /restart  GET /status  GET /task[/<id>]")

    server = HTTPServer((args.host, args.port), PdControlHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log_default("服务器停止。")


if __name__ == "__main__":
    main()
