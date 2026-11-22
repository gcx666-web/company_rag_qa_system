# -*- coding: utf-8 -*-
"""
desp.py —— 依赖注入层：向量模型 / 文档存储 / 检索服务 / LLM 实例。

本文件是“客户端复用”的落点：原来每个请求都会 new 一次
``OpenAIEmbeddings``、``ChromaVectorStore``、``DocumentStore``、``LLM``，
其中 Chroma 客户端初始化、OpenAI 连接池创建都有实打实的开销。现在改为：

- 向量模型：交给 ``src.embeddings.factory`` 做进程内单例；
- Chroma 客户端：按持久化目录复用（``chroma_client.get_chroma_client``）；
- DocumentStore / SearchService：按集合名缓存，集合数量通常远小于请求数；
- LLM：按“供应商 + 模型 + 地址”缓存，复用一个 OpenAI 客户端。

所有缓存都带 ``threading.RLock``，并发请求只会触发一次初始化。
"""

from __future__ import annotations

import logging
import threading

from src.api.config import settings
from src.database.chroma_client import get_chroma_client
from src.database.document_store import DocumentStore
from src.embeddings.base import Base
from src.embeddings.chorm_store import ChromaVectorStore
from src.embeddings.factory import get_embedding_model as _factory_embedding_model
from src.llm.llm import LLM
from src.retrieval.search_service import SearchService

logger = logging.getLogger(__name__)

__all__ = [
    "get_embedding_model",
    "get_document_store",
    "get_search_service",
    "get_llm",
    "evict_document_store",
    "reset_document_store_cache",
]

# 文档存储 / 检索服务缓存：collection_name -> 实例
_STORE_LOCK = threading.RLock()
_DOCUMENT_STORES: dict[str, DocumentStore] = {}
_SEARCH_SERVICES: dict[str, SearchService] = {}

# LLM 实例缓存：provider|model|base_url -> LLM
_LLM_LOCK = threading.RLock()
_LLMS: dict[str, LLM] = {}


def get_embedding_model() -> Base:
    """获取进程级复用的向量模型（自动处理在线/离线与缓存包装）。"""
    return _factory_embedding_model()


def get_document_store(collection_name: str) -> DocumentStore:
    """获取（缓存的）文档存储实例。

    :param collection_name: Chroma 集合名称，同时作为写入/检索的 namespace。
    """
    key = str(collection_name)
    with _STORE_LOCK:
        store = _DOCUMENT_STORES.get(key)
        if store is None:
            # 复用 Chroma 客户端，避免每次请求重新打开持久化目录
            vector_store = ChromaVectorStore(
                collection_name=key,
                persist_directory=settings.CHROMA_PERSIST_DIR,
                distance=settings.CHROMA_DISTANCE,
                client=get_chroma_client(settings.CHROMA_PERSIST_DIR),
            )
            store = DocumentStore(
                embedding_model=get_embedding_model(),
                vector_store=vector_store,
                collection_name=key,
                persist_directory=settings.CHROMA_PERSIST_DIR,
            )
            _DOCUMENT_STORES[key] = store
            logger.info("文档存储实例已缓存：collection=%s", key)
        return store


def get_search_service(collection_name: str) -> SearchService:
    """获取（缓存的）检索服务实例。"""
    key = str(collection_name)
    with _STORE_LOCK:
        service = _SEARCH_SERVICES.get(key)
        if service is None:
            service = SearchService(document_store=get_document_store(key))
            _SEARCH_SERVICES[key] = service
        return service


def get_llm(model: str) -> LLM:
    """获取（缓存的）LLM 实例。

    :param model: 供应商名称（zhipu / qwen / siliconflow）。
    :raises ValueError: 供应商不存在或未配置密钥。
    """
    if model not in settings.LLM_PROVIDER:
        raise ValueError(f"不支持的 LLM 供应商：{model}")

    config = settings.LLM_PROVIDER[model] or {}
    api_key = config.get("API_KEY") or ""
    base_url = config.get("BASE_URL") or ""
    model_name = config.get("MODEL") or ""
    if not api_key:
        raise ValueError(
            f"LLM 供应商 {model} 未配置密钥，请在 .env 中设置 "
            f"{model.upper()}_API_KEY（或改用其它已配置的供应商）"
        )

    key = f"{model}|{model_name}|{base_url}"
    with _LLM_LOCK:
        llm = _LLMS.get(key)
        if llm is None:
            llm = LLM(
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                timeout=settings.LLM_TIMEOUT_SECONDS,
            )
            _LLMS[key] = llm
            logger.info("LLM 客户端已缓存：provider=%s, model=%s", model, model_name)
        return llm


def evict_document_store(collection_name: str) -> None:
    """从缓存中移除单个集合的存储/检索实例（删除集合后调用）。

    注意：只移除缓存引用，不会操作 Chroma；集合本身由调用方删除。
    这样下次再访问同名集合时会重新构造，不会拿到已被删除的旧句柄。
    """
    key = str(collection_name)
    with _STORE_LOCK:
        _DOCUMENT_STORES.pop(key, None)
        _SEARCH_SERVICES.pop(key, None)
    logger.info("文档存储缓存已移除：collection=%s", key)


def reset_document_store_cache() -> None:
    """清空文档存储与检索服务缓存（删除集合/切换向量库后调用）。"""
    with _STORE_LOCK:
        _DOCUMENT_STORES.clear()
        _SEARCH_SERVICES.clear()
    logger.info("文档存储缓存已重置")
