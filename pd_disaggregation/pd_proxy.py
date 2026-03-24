#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PD 分离负载均衡代理。

基于 vLLM disagg_proxy_demo.py 精简而来，支持：
  - 多 Prefill / Decode 实例轮询调度
  - /v1/models 返回代理列出的模型名
  - /v1/completions 与 /v1/chat/completions
  - /status 查看集群状态
  - /health 健康检查

用法：
  python pd_proxy.py --model <model_name> \
       --prefill localhost:9000 --decode localhost:9010 localhost:9011 \
       --port 8000
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional

import aiohttp
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
logger = logging.getLogger("pd_proxy")
logging.basicConfig(level=logging.INFO)


class SchedulingPolicy(ABC):
    @abstractmethod
    def schedule(self, cycler: itertools.cycle):
        raise NotImplementedError


class RoundRobinSchedulingPolicy(SchedulingPolicy):
    def schedule(self, cycler: itertools.cycle) -> str:
        return next(cycler)


class Proxy:
    def __init__(
        self,
        prefill_instances: list[str],
        decode_instances: list[str],
        model: str,
        scheduling_policy: SchedulingPolicy | None = None,
    ):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.model = model
        self.scheduling_policy = scheduling_policy or RoundRobinSchedulingPolicy()
        self.router = APIRouter()
        self._setup_routes()

    def _setup_routes(self):
        self.router.post(
            "/v1/completions", dependencies=[Depends(self._validate_json)]
        )(self.create_completion)
        self.router.post(
            "/v1/chat/completions", dependencies=[Depends(self._validate_json)]
        )(self.create_chat_completion)
        self.router.get("/v1/models", response_class=JSONResponse)(self.list_models)
        self.router.get("/status", response_class=JSONResponse)(self.get_status)
        self.router.get("/health", response_class=JSONResponse)(self.health)

    async def _validate_json(self, raw_request: Request):
        content_type = raw_request.headers.get("content-type", "").lower()
        if content_type != "application/json":
            raise HTTPException(status_code=415, detail="Only application/json allowed")

    def _schedule(self, cycler: itertools.cycle) -> str:
        return self.scheduling_policy.schedule(cycler)

    async def _forward_request(self, url, data, use_chunked=True):
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"}
            try:
                async with session.post(url=url, json=data, headers=headers) as response:
                    if 200 <= response.status < 300 or 400 <= response.status < 500:
                        if use_chunked:
                            async for chunk in response.content.iter_chunked(1024):
                                yield chunk
                        else:
                            yield await response.read()
                    else:
                        error_content = await response.text()
                        logger.error("Request failed %s: %s", response.status, error_content)
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Upstream error {response.status}: {error_content}",
                        )
            except aiohttp.ClientError as e:
                logger.error("ClientError: %s", e)
                raise HTTPException(status_code=502, detail="Bad Gateway") from e

    def _remove_instance(self, instance_type: str, instance: str):
        if instance_type == "decode" and instance in self.decode_instances:
            self.decode_instances.remove(instance)
            self.decode_cycler = itertools.cycle(self.decode_instances)
        elif instance_type == "prefill" and instance in self.prefill_instances:
            self.prefill_instances.remove(instance)
            self.prefill_cycler = itertools.cycle(self.prefill_instances)

    async def create_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            # Prefill stage: max_tokens=1 to trigger KV cache transfer
            kv_prepare = request.copy()
            kv_prepare["max_tokens"] = 1

            prefill = self._schedule(self.prefill_cycler)
            try:
                async for _ in self._forward_request(
                    f"http://{prefill}/v1/completions", kv_prepare
                ):
                    continue
            except HTTPException as e:
                self._remove_instance("prefill", prefill)
                raise e

            # Decode stage
            decode = self._schedule(self.decode_cycler)
            try:
                generator = self._forward_request(
                    f"http://{decode}/v1/completions", request
                )
            except HTTPException as e:
                self._remove_instance("decode", decode)
                raise e

            return StreamingResponse(generator)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error in create_completion")
            raise HTTPException(status_code=500, detail="Internal proxy error")

    async def create_chat_completion(self, raw_request: Request):
        try:
            request = await raw_request.json()

            kv_prepare = request.copy()
            kv_prepare["max_tokens"] = 1
            if "max_completion_tokens" in kv_prepare:
                kv_prepare["max_completion_tokens"] = 1

            prefill = self._schedule(self.prefill_cycler)
            try:
                async for _ in self._forward_request(
                    f"http://{prefill}/v1/chat/completions", kv_prepare
                ):
                    continue
            except HTTPException as e:
                self._remove_instance("prefill", prefill)
                raise e

            decode = self._schedule(self.decode_cycler)
            try:
                generator = self._forward_request(
                    f"http://{decode}/v1/chat/completions", request
                )
            except HTTPException as e:
                self._remove_instance("decode", decode)
                raise e

            return StreamingResponse(content=generator)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Error in create_chat_completion")
            raise HTTPException(status_code=500, detail="Internal proxy error")

    async def list_models(self):
        """兼容 OpenAI /v1/models 接口，返回代理配置的模型名。"""
        import time as _time
        return {
            "object": "list",
            "data": [
                {
                    "id": self.model,
                    "object": "model",
                    "created": int(_time.time()),
                    "owned_by": "pd-proxy",
                }
            ],
        }

    async def get_status(self):
        return {
            "prefill_count": len(self.prefill_instances),
            "decode_count": len(self.decode_instances),
            "prefill_nodes": self.prefill_instances,
            "decode_nodes": self.decode_instances,
        }

    async def health(self):
        return {"status": "ok"}


def create_app(
    prefill_instances: list[str],
    decode_instances: list[str],
    model: str,
) -> FastAPI:
    """创建 FastAPI app（供 pd_service_ctl 直接调用）。"""
    app = FastAPI(title="PD Proxy")
    proxy = Proxy(prefill_instances, decode_instances, model)
    app.include_router(proxy.router)
    return app


def run_proxy(
    prefill_instances: list[str],
    decode_instances: list[str],
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    """一键启动代理服务（阻塞）。"""
    app = create_app(prefill_instances, decode_instances, model)
    config = uvicorn.Config(app, host=host, port=port, loop="uvloop")
    server = uvicorn.Server(config)
    server.run()


def parse_args():
    parser = argparse.ArgumentParser(description="PD disaggregated proxy server")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model name")
    parser.add_argument(
        "--prefill", "-p", type=str, nargs="+", required=True,
        help="Prefill instance URLs (host:port)",
    )
    parser.add_argument(
        "--decode", "-d", type=str, nargs="+", required=True,
        help="Decode instance URLs (host:port)",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_proxy(
        prefill_instances=args.prefill,
        decode_instances=args.decode,
        model=args.model,
        host=args.host,
        port=args.port,
    )
