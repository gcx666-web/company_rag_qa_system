# -*- coding: utf-8 -*-
"""
system_router.py —— 系统状态 / 向量模型管理接口。

前端“系统状态”页依赖这些接口展示：

- 当前向量模式（在线 / 离线 / 自动解析结果）、模型名、维度、是否已加载；
- 向量缓存命中率、内存条数、磁盘条数、磁盘占用；
- 各 LLM 供应商是否已配置、默认供应商；
- 存储目录、去重开关等功能开关；
- 预热向量模型、清空向量缓存两个运维动作。

所有耗时操作（尤其离线模型首次加载）都通过 run_in_threadpool 执行，避免
阻塞 FastAPI 事件循环。
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from src.api.config import settings
from src.api.models import CacheClearResponse, WarmupResponse
from src.database.chroma_client import list_collection_names
from src.database.upload_registry import get_upload_registry
from src.embeddings.embedding_cache import get_default_embedding_cache
from src.embeddings.factory import (
    clear_embedding_cache,
    describe_embedding,
    warmup_embedding,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["系统状态"])


def _default_llm_provider() -> str | None:
    """解析默认 LLM 供应商。

    ``DEFAULT_LLM_MODEL`` 可能写成供应商名（zhipu/qwen/siliconflow），也可能写成
    具体模型名；两种写法都要兼容。都没有配置时，退回第一个有密钥的供应商。
    """
    raw = str(settings.Default_LLM_MODEL or "").strip()
    if raw in settings.LLM_PROVIDER:
        return raw
    for name, config in settings.LLM_PROVIDER.items():
        if (config or {}).get("MODEL") == raw and raw:
            return name
    for name, config in settings.LLM_PROVIDER.items():
        if (config or {}).get("API_KEY"):
            return name
    return None


def _llm_providers() -> list[dict]:
    """返回前端可用的 LLM 供应商列表（不暴露密钥）。"""
    default = _default_llm_provider()
    providers = []
    for name, config in settings.LLM_PROVIDER.items():
        config = config or {}
        providers.append(
            {
                "name": name,
                "model": config.get("MODEL") or "",
                "base_url": config.get("BASE_URL") or "",
                "configured": bool(config.get("API_KEY")),
                "is_default": name == default,
            }
        )
    return providers


@router.get("/status", summary="系统总览")
def system_status() -> dict:
    """返回系统总览：向量模型、缓存、LLM、存储与功能开关。"""
    embedding = describe_embedding()

    registry = get_upload_registry()
    try:
        registry_count = registry.count()
    except Exception:
        registry_count = 0

    try:
        collections = list_collection_names(settings.CHROMA_PERSIST_DIR)
    except Exception:
        collections = []

    return {
        "app": {
            "name": "company-rag-qa-system",
            "version": "0.2.0",
        },
        "embedding": embedding,
        "llm": {
            "default_provider": _default_llm_provider(),
            "default_model": settings.Default_LLM_MODEL,
            "providers": _llm_providers(),
            "temperature": settings.LLM_TEMPERATURE,
            "max_tokens": settings.LLM_MAX_TOKENS,
        },
        "storage": {
            "chroma_dir": settings.CHROMA_PERSIST_DIR,
            "upload_dir": settings.UPLOAD_DIR,
            "registry_db": settings.UPLOAD_REGISTRY_DB,
            "cache_dir": settings.EMBEDDING_CACHE_DIR,
            "collections": len(collections),
            "registry_records": registry_count,
            "dedup_enabled": settings.UPLOAD_DEDUP_ENABLED,
        },
        "features": {
            "embedding_cache": settings.EMBEDDING_CACHE_ENABLED,
            "offline_mode": embedding.get("mode") == "offline",
            "duplicate_detection": settings.UPLOAD_DEDUP_ENABLED,
            "client_reuse": True,
            "parser_cache": True,
        },
        "environment": {
            "offline_env": {
                name: os.getenv(name)
                for name in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "EMBEDDING_MODE",
                    "EMBEDDING_OFFLINE",
                )
            }
        },
    }


@router.get("/embedding/status", summary="向量模型状态")
def embedding_status() -> dict:
    """只返回向量模型与缓存状态（系统页自动刷新时使用，响应更小）。"""
    return describe_embedding()


@router.get("/embedding/cache", summary="向量缓存统计")
def embedding_cache_stats() -> dict:
    """返回向量缓存统计信息。"""
    return get_default_embedding_cache().stats()


@router.post(
    "/embedding/warmup",
    response_model=WarmupResponse,
    summary="预热向量模型",
)
async def embedding_warmup() -> WarmupResponse:
    """加载/调用一次向量模型，验证配置可用并提前建立连接/缓存。

    离线模式下首次调用会加载本地 SentenceTransformer，可能耗时数秒到数十秒。
    """
    try:
        result = await run_in_threadpool(warmup_embedding)
    except Exception as exc:
        logger.exception("向量模型预热失败")
        raise HTTPException(
            status_code=500, detail=f"向量模型预热失败: {exc}"
        ) from exc
    return WarmupResponse(**result)


@router.post(
    "/embedding/cache/clear",
    response_model=CacheClearResponse,
    summary="清空向量缓存",
)
def embedding_cache_clear() -> CacheClearResponse:
    """清空向量缓存（不会删除向量库里的文档）。"""
    stats = clear_embedding_cache()
    return CacheClearResponse(cleared=True, cache=stats)
