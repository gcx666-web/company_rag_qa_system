# -*- coding: utf-8 -*-
"""
chroma_client.py —— Chroma 持久化客户端复用。

为什么需要单独缓存客户端
------------------------
``chromadb.PersistentClient(path=...)`` 每次构造都会重新初始化底层系统、读取
持久化目录并建立 SQLite 连接。原实现里“每个上传/问答请求都 new 一次
ChromaVectorStore”，等于每个请求都重建一次客户端，文件越多、并发越高，
初始化开销越明显。

这里按“持久化目录的绝对路径”缓存客户端：

- 同一进程内不同集合共享同一个 PersistentClient；
- 线程安全（RLock 保护）；
- 提供集合枚举/删除工具，供知识库管理接口使用；
- 测试或需要重新连接时调用 ``reset_chroma_clients()``。

注意：``ChromaVectorStore`` 仍然支持不传 client 的旧用法，方便单元测试
mock ``chromadb.PersistentClient``；正式运行链路由依赖注入层传入缓存客户端。
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

import chromadb

logger = logging.getLogger(__name__)

__all__ = [
    "get_chroma_client",
    "list_collection_names",
    "delete_collection",
    "reset_chroma_clients",
]

_CLIENT_LOCK = threading.RLock()
_CLIENTS: dict[str, Any] = {}


def _normalize_path(persist_directory: Optional[str]) -> str:
    """把持久化目录归一化成绝对路径，避免相对路径因工作目录不同而重复建客户端。"""
    raw = str(persist_directory or "./data/chroma_db")
    return str(Path(raw).expanduser().resolve())


def get_chroma_client(persist_directory: Optional[str] = None):
    """获取（并缓存）指定持久化目录的 Chroma 客户端。

    :param persist_directory: 向量库目录；None 时使用默认目录。
    :return: ``chromadb.PersistentClient`` 实例。
    """
    path = _normalize_path(persist_directory)
    with _CLIENT_LOCK:
        client = _CLIENTS.get(path)
        if client is None:
            Path(path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=path)
            _CLIENTS[path] = client
            logger.info("Chroma 客户端已创建并缓存：%s", path)
        return client


def list_collection_names(persist_directory: Optional[str] = None) -> list[str]:
    """列出持久化目录下的所有集合名。

    Chroma 不同版本 ``list_collections()`` 的返回类型不同（str 或 Collection），
    这里统一兼容，只返回名称字符串。
    """
    client = get_chroma_client(persist_directory)
    names: list[str] = []
    try:
        collections = client.list_collections()
    except Exception as exc:
        logger.warning("列出 Chroma 集合失败：%s", exc)
        return names

    for item in collections or []:
        name = item if isinstance(item, str) else getattr(item, "name", None)
        if name:
            names.append(str(name))
    # 去重并排序，保证前端展示顺序稳定
    return sorted(set(names))


def delete_collection(
    collection_name: str, persist_directory: Optional[str] = None
) -> bool:
    """删除指定集合；集合不存在时返回 False（不抛异常）。"""
    client = get_chroma_client(persist_directory)
    try:
        client.delete_collection(name=collection_name)
        return True
    except Exception as exc:
        logger.warning("删除 Chroma 集合 %s 失败：%s", collection_name, exc)
        return False


def reset_chroma_clients() -> None:
    """清空客户端缓存（测试或切换持久化目录后调用）。"""
    with _CLIENT_LOCK:
        _CLIENTS.clear()
    logger.info("Chroma 客户端缓存已重置")
