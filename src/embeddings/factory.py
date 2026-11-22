# -*- coding: utf-8 -*-
"""
factory.py —— 向量模型工厂（在线 / 离线 / 自动切换 + 单例复用 + 缓存包装）。

核心目标
--------
1. **客户端只创建一次**：每个“模式 + 模型 + 维度/设备”组合只构造一个模型实例，
   其内部 OpenAI 客户端或 SentenceTransformer 也随之复用，避免每个请求都
   重新 new 客户端；
2. **离线模式**：``EMBEDDING_MODE=offline`` 时使用本地 SentenceTransformer，
   并设置 HuggingFace 离线变量，整个链路不需要网络；
3. **自动降级**：``EMBEDDING_MODE=auto`` 时，有在线密钥就走在线接口，没有密钥
   自动使用本地模型，开发/内网环境不用改代码；
4. **统一缓存**：所有模型实例外层都包 ``CachedEmbeddings``，共享同一份
   SQLite 向量缓存。

对外主要接口
------------adsTR^&tr6aDS
- ``get_embedding_model()``：获取（缓存的）向量模型实例；
- ``describe_embedding()``：获取模式、模型、维度、缓存命中率等状态；
- ``warmup_embedding()``：预热模型，返回维度与耗时；
- ``clear_embedding_cache()``：清空向量缓存；
- ``reset_embedding_model()``：丢弃实例（测试或热切换配置时使用）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

from src.api.config import settings
from src.embeddings.base import Base
from src.embeddings.embedding_cache import (
    CachedEmbeddings,
    get_default_embedding_cache,
)
from src.embeddings.openai_enbeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

__all__ = [
    "resolve_embedding_mode",
    "get_embedding_model",
    "describe_embedding",
    "warmup_embedding",
    "clear_embedding_cache",
    "reset_embedding_model",
]

# 合法的模式值
_VALID_MODES = {"auto", "online", "offline"}

# 实例缓存：key -> 模型对象；用 RLock 保证并发请求只构造一次
_MODEL_LOCK = threading.RLock()
_MODEL_INSTANCES: dict[str, Base] = {}

# 离线模式相关的环境变量，状态接口会把它们原样展示出来，便于排查
_OFFLINE_ENV_KEYS = (
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
    "HF_ENDPOINT",
    "HF_HOME",
)


def _is_truthy(value: Optional[str]) -> bool:
    """把常见“真值”环境变量写法转成 bool。"""
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_embedding_mode() -> str:
    """解析最终生效的向量模式：auto / online / offline。

    - ``EMBEDDING_MODE=offline``：无条件离线；
    - ``EMBEDDING_MODE=online``：无条件在线（缺少密钥时抛错）；
    - ``EMBEDDING_MODE=auto``：有密钥且未设置 ``EMBEDDING_OFFLINE=1`` 时在线，
      否则离线。

    :raises ValueError: 配置了非法模式值时抛出。
    """
    configured = (settings.EMBEDDING_MODE or "auto").strip().lower()
    if configured not in _VALID_MODES:
        raise ValueError(
            f"EMBEDDING_MODE 配置非法：{configured!r}，"
            f"可选值为 {sorted(_VALID_MODES)}"
        )
    if configured == "offline":
        return "offline"
    if configured == "online":
        return "online"
    # auto：显式 EMBEDDING_OFFLINE=1 优先，其次看是否配置了在线密钥
    if _is_truthy(os.getenv("EMBEDDING_OFFLINE")):
        return "offline"
    return "online" if settings.EMBEDDING_API_KEY else "offline"


def _offline_model_name() -> str:
    """离线模式下实际使用的模型名 / 本地目录。"""
    local_dir = str(settings.OFFLINE_EMBEDDING_DIR or "").strip()
    if local_dir:
        # 目录不存在时给出 warning，但仍交给 SentenceTransformer 处理，
        # 以便支持 hf 模型名与本地目录两种写法
        if not os.path.isdir(local_dir):
            logger.warning(
                "OFFLINE_EMBEDDING_DIR 指向的目录不存在：%s，将回退到模型名 %s",
                local_dir,
                settings.OFFLINE_EMBEDDING_MODEL,
            )
        else:
            return local_dir
    return settings.OFFLINE_EMBEDDING_MODEL


def _build_online_model() -> Base:
    """构造在线 OpenAI 兼容向量模型（密钥只读取一次，客户端随实例复用）。"""
    if not settings.EMBEDDING_API_KEY:
        raise ValueError(
            "当前为在线向量模式，但未配置密钥：请在 .env 中设置 "
            "EMBEDDING_API_KEY（或 SILICONFLOW_API_KEY），"
            "或把 EMBEDDING_MODE 改为 offline / auto"
        )
    logger.info(
        "使用在线向量模型：%s（base_url=%s）",
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_BASE_URL,
    )
    return OpenAIEmbeddings(
        api_key=settings.EMBEDDING_API_KEY,
        model=settings.EMBEDDING_MODEL,
        base_url=settings.EMBEDDING_BASE_URL,
        dimension=settings.EMBEDDING_DIMENSION,
    )


def _build_offline_model() -> Base:
    """构造本地离线向量模型（懒加载，避免启动时阻塞）。"""
    # 延迟导入：在线模式下完全不加载 sentence-transformers，减少启动开销
    from src.embeddings.local_embeddings import LocalEmbeddings

    model_name = _offline_model_name()
    logger.info(
        "使用离线向量模型：%s（device=%s）",
        model_name,
        settings.OFFLINE_EMBEDDING_DEVICE,
    )
    return LocalEmbeddings(
        model_name=model_name,
        device=settings.OFFLINE_EMBEDDING_DEVICE,
        lazy=True,
        offline=True,
        local_files_only=settings.OFFLINE_EMBEDDING_LOCAL_ONLY,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


def _instance_key(mode: str) -> str:
    """生成实例缓存 key：不同模式 / 模型 / 维度必须使用不同实例。"""
    if mode == "offline":
        return (
            f"offline|{_offline_model_name()}|{settings.OFFLINE_EMBEDDING_DEVICE}"
        )
    return (
        f"online|{settings.EMBEDDING_MODEL}|{settings.EMBEDDING_BASE_URL}|"
        f"{settings.EMBEDDING_DIMENSION or 'default'}"
    )


def _wrap_with_cache(inner: Base, mode: str) -> Base:
    """给真实模型包一层缓存；缓存名称带上模式与模型，避免串用不同模型的向量。"""
    cache = get_default_embedding_cache()
    # 在线模式维度可提前确定，把它纳入缓存标识，避免改维度后命中旧维度向量
    suffix = ""
    if mode == "online":
        try:
            suffix = f"|{inner.get_dimension()}"
        except Exception:
            suffix = ""
    cache_name = f"{mode}|{inner.get_model_name()}{suffix}"
    return CachedEmbeddings(
        inner=inner,
        cache=cache,
        cache_name=cache_name,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
    )


def get_embedding_model(force_reload: bool = False) -> Base:
    """获取进程级复用的向量模型实例。

    :param force_reload: 为 True 时丢弃旧实例并重新构造（测试/热切换用）。
    :raises ValueError: 在线模式缺少密钥、模式配置非法等。
    """
    mode = resolve_embedding_mode()
    key = _instance_key(mode)

    with _MODEL_LOCK:
        if force_reload:
            _MODEL_INSTANCES.pop(key, None)
        instance = _MODEL_INSTANCES.get(key)
        if instance is None:
            started = time.perf_counter()
            inner = _build_offline_model() if mode == "offline" else _build_online_model()
            instance = _wrap_with_cache(inner, mode)
            _MODEL_INSTANCES[key] = instance
            logger.info(
                "向量模型实例已创建并缓存：key=%s（耗时 %.1f ms）",
                key,
                (time.perf_counter() - started) * 1000,
            )
        return instance


def _inner_model(model: Base) -> Base:
    """取出缓存包装器内部的真实模型。"""
    return getattr(model, "inner", model)


def describe_embedding() -> dict:
    """返回向量模型与缓存的状态信息，供前端系统页展示。

    该函数不会主动加载离线模型（除非已经加载过），因此可以安全地随页面刷新
    高频调用。
    """
    info: dict[str, Any] = {
        "configured_mode": settings.EMBEDDING_MODE,
        "mode": "unknown",
        "provider": "unknown",
        "model": "",
        "dimension": None,
        "loaded": False,
        "available": False,
        "base_url": None,
        "batch_size": settings.EMBEDDING_BATCH_SIZE,
        "cache": {},
        "offline": {
            "model": settings.OFFLINE_EMBEDDING_MODEL,
            "dir": settings.OFFLINE_EMBEDDING_DIR or None,
            "device": settings.OFFLINE_EMBEDDING_DEVICE,
            "local_files_only": settings.OFFLINE_EMBEDDING_LOCAL_ONLY,
            "env": {name: os.getenv(name) for name in _OFFLINE_ENV_KEYS},
        },
        "error": None,
    }

    # 先填充缓存统计，即使模型构造失败也能看到缓存状态
    try:
        info["cache"] = get_default_embedding_cache().stats()
    except Exception as exc:  # 缓存故障不应影响状态接口
        info["cache"] = {"enabled": False, "error": str(exc)}

    try:
        mode = resolve_embedding_mode()
        info["mode"] = mode
        model = get_embedding_model()
        inner = _inner_model(model)
        info["available"] = True

        if mode == "offline":
            info["provider"] = "local"
            info["model"] = inner.get_model_name()
            # LocalEmbeddings 提供 info()，不会触发懒加载
            local_info = getattr(inner, "info", None)
            if callable(local_info):
                details = local_info()
                info["loaded"] = bool(details.get("loaded"))
                # 模型未加载时用配置的维度提示值，保证知识库兼容性检查在
                # 启动后第一次打开页面就能生效（无需先触发一次向量化）。
                info["dimension"] = (
                    details.get("dimension")
                    or settings.OFFLINE_EMBEDDING_DIMENSION
                )
            stats = getattr(model, "cache_stats", None)
            if callable(stats):
                info["cache"] = stats()
        else:
            info["provider"] = "openai-compatible"
            info["model"] = inner.get_model_name()
            info["base_url"] = getattr(inner, "base_url", None)
            info["loaded"] = True
            info["dimension"] = inner.get_dimension()
            stats = getattr(model, "cache_stats", None)
            if callable(stats):
                info["cache"] = stats()
    except Exception as exc:
        # 配置缺失（例如在线模式没密钥）不让页面整体 500，而是展示错误原因
        info["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("读取向量模型状态失败：%s", exc)

    return info


def warmup_embedding() -> dict:
    """预热向量模型：执行一次真实向量化并返回维度与耗时。

    离线模式下这会触发模型加载；在线模式下会真实调用一次接口。
    结果会被缓存，因此重复预热几乎瞬间完成。
    """
    model = get_embedding_model()
    started = time.perf_counter()
    vector = model.embed_text("向量模型预热：公司知识库问答")
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not vector:
        raise RuntimeError("向量模型预热失败：返回空向量")
    return {
        "status": "ok",
        "mode": resolve_embedding_mode(),
        "model": model.get_model_name(),
        "dimension": len(vector),
        "elapsed_ms": round(elapsed_ms, 2),
        "cache": get_default_embedding_cache().stats(),
    }


def clear_embedding_cache() -> dict:
    """清空向量缓存（不影响向量库中的文档向量）。"""
    cache = get_default_embedding_cache()
    stats = cache.clear()
    logger.info("向量缓存已清空：%s", stats)
    return stats


def reset_embedding_model() -> None:
    """丢弃已缓存的模型实例（测试或配置热切换时使用）。"""
    with _MODEL_LOCK:
        _MODEL_INSTANCES.clear()
    logger.info("向量模型实例缓存已重置")
