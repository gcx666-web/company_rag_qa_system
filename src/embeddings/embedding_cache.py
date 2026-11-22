# -*- coding: utf-8 -*-
"""
embedding_cache.py —— 向量结果缓存（内存 LRU + SQLite 磁盘持久化）。

为什么需要它
------------
RAG 系统里最贵、最慢的一步通常是“文本 -> 向量”：

1. 在线接口按 token 计费、有网络延迟，同一段文本在重复上传、重复提问时会被
   反复向量化；
2. 本地离线模型（SentenceTransformer）推理也耗时，重复分块会重复计算。

本模块在 ``Base`` 向量模型外面包一层 ``CachedEmbeddings``：

- 计算缓存键：``sha256(模型标识 + "\\x00" + 文本)``，相同文本 + 相同模型必然命中；
- 一级缓存：进程内 ``OrderedDict`` 实现的 LRU，命中时零 IO；
- 二级缓存：SQLite 文件，进程重启后仍然有效，多进程部署也能共享；
- 批量接口 ``embed_batch`` 只把“未命中”的文本发给真实模型，再按原顺序拼回；
- 缓存向量用 float32 二进制存储，比 JSON 文本更省空间、读写更快。

线程安全
--------
所有内存与磁盘操作都持有同一把 ``RLock``；SQLite 连接使用
``check_same_thread=False``，配合锁串行化访问，避免多线程写入冲突。
"""

from __future__ import annotations

import hashlib
import logging
import math
import sqlite3
import threading
import time
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from src.embeddings.base import Base

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingCache",
    "CachedEmbeddings",
    "get_default_embedding_cache",
    "reset_default_embedding_cache",
]


class EmbeddingCache:
    """向量缓存：内存 LRU + SQLite 磁盘持久化。

    :param db_path:          SQLite 文件路径；传 None 表示只使用内存缓存。
    :param max_memory_items: 内存中最多保留的向量条数（0 表示禁用内存缓存）。
    :param ttl_seconds:      磁盘缓存有效期（秒），0 表示永不过期。
    :param enabled:          总开关；关闭后 get 恒为未命中、put 不写入。
    """

    # 磁盘表结构：cache_key 是“模型 + 文本”的摘要，vector 用 float32 二进制保存
    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS embeddings (
        cache_key   TEXT PRIMARY KEY,
        model       TEXT NOT NULL,
        text_hash   TEXT NOT NULL,
        dimension   INTEGER NOT NULL,
        vector      BLOB NOT NULL,
        created_at  REAL NOT NULL,
        last_access REAL NOT NULL,
        hits        INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
    CREATE INDEX IF NOT EXISTS idx_embeddings_last_access ON embeddings(last_access);
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_memory_items: int = 2048,
        ttl_seconds: int = 0,
        enabled: bool = True,
    ) -> None:
        self.db_path = str(db_path) if db_path else None
        self.max_memory_items = max(0, int(max_memory_items))
        self.ttl_seconds = max(0, int(ttl_seconds))
        self.enabled = bool(enabled)

        # 内存缓存：key -> (向量, 最近访问时间)
        self._memory: "OrderedDict[str, tuple[List[float], float]]" = OrderedDict()
        self._lock = threading.RLock()

        # SQLite 连接延迟创建：只走内存缓存或磁盘不可用时不影响主流程
        self._conn: Optional[sqlite3.Connection] = None
        self._disk_ready = False
        self._disk_broken = False

        # 运行统计（供 /api/system 展示命中率）
        self._hits = 0
        self._misses = 0
        self._memory_hits = 0
        self._disk_hits = 0
        self._writes = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    @staticmethod
    def make_key(model_name: str, text: str) -> str:
        """生成缓存键：模型名 + 文本的 SHA-256 摘要。

        加入模型名是为了避免“换了向量模型却命中旧模型向量”的严重问题。
        """
        payload = f"{model_name}\x00{text}".encode("utf-8", errors="replace")
        return hashlib.sha256(payload).hexdigest()

    def get(self, model_name: str, text: str) -> Optional[List[float]]:
        """读取单个文本的缓存向量；未命中返回 None。"""
        if not self.enabled:
            return None
        key = self.make_key(model_name, text)
        with self._lock:
            return self._get_locked(key)

    def get_many(
        self, model_name: str, texts: Sequence[str]
    ) -> List[Optional[List[float]]]:
        """批量读取缓存，返回与 texts 等长的列表；未命中的位置为 None。

        整批只加一次锁，避免几百个分块反复抢锁。
        """
        if not self.enabled:
            return [None for _ in texts]
        with self._lock:
            return [self._get_locked(self.make_key(model_name, text)) for text in texts]

    def put_many(
        self,
        model_name: str,
        texts: Sequence[str],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        """批量写入缓存（内存 + 磁盘）。长度不一致或空向量会被跳过。"""
        if not self.enabled or not texts:
            return
        if len(texts) != len(vectors):
            logger.warning(
                "缓存写入数量不一致：%d 条文本 vs %d 条向量，已跳过",
                len(texts),
                len(vectors),
            )
            return

        now = time.time()
        rows: list[tuple] = []
        with self._lock:
            for text, raw_vector in zip(texts, vectors):
                if raw_vector is None:
                    continue
                vector = [float(value) for value in raw_vector]
                if not vector or not all(math.isfinite(value) for value in vector):
                    # NaN/Inf 写进向量库会让检索彻底失效，宁可放弃缓存
                    logger.warning("跳过包含 NaN/Inf 或空值的向量缓存")
                    continue
                key = self.make_key(model_name, text)
                self._put_memory_locked(key, vector, now)
                text_hash = hashlib.sha256(
                    str(text).encode("utf-8", errors="replace")
                ).hexdigest()
                rows.append(
                    (
                        key,
                        model_name,
                        text_hash,
                        len(vector),
                        array("f", vector).tobytes(),
                        now,
                        now,
                    )
                )

            if rows:
                self._writes += len(rows)
                self._put_disk_locked(rows)

    def stats(self) -> dict:
        """返回缓存统计信息，供前端“系统状态”页面展示。"""
        with self._lock:
            total = self._hits + self._misses
            memory_items = len(self._memory)
            disk_entries, disk_bytes = self._disk_stats_locked()
            return {
                "enabled": self.enabled,
                "db_path": self.db_path,
                "memory_items": memory_items,
                "memory_capacity": self.max_memory_items,
                "disk_entries": disk_entries,
                "disk_bytes": disk_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "memory_hits": self._memory_hits,
                "disk_hits": self._disk_hits,
                "writes": self._writes,
                "errors": self._errors,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "ttl_seconds": self.ttl_seconds,
            }

    def clear(self) -> dict:
        """清空内存与磁盘缓存，返回清空后的统计信息。"""
        with self._lock:
            self._memory.clear()
            if self._ensure_disk_locked():
                try:
                    assert self._conn is not None
                    self._conn.execute("DELETE FROM embeddings")
                    self._conn.commit()
                except sqlite3.Error as exc:
                    self._errors += 1
                    logger.warning("清空磁盘向量缓存失败：%s", exc)
            self._hits = 0
            self._misses = 0
            self._memory_hits = 0
            self._disk_hits = 0
            self._writes = 0
            return self.stats()

    def close(self) -> None:
        """关闭 SQLite 连接（进程退出或测试清理时调用）。"""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None
                self._disk_ready = False

    # ------------------------------------------------------------------
    # 内部工具：内存
    # ------------------------------------------------------------------
    def _get_locked(self, key: str) -> Optional[List[float]]:
        """带锁的缓存读取：先查内存，再查磁盘，并把磁盘命中回填内存。"""
        now = time.time()

        # 1) 内存 LRU
        item = self._memory.get(key)
        if item is not None:
            vector, accessed_at = item
            if self._is_expired(accessed_at, now):
                self._memory.pop(key, None)
            else:
                self._memory.move_to_end(key)
                self._hits += 1
                self._memory_hits += 1
                return list(vector)

        # 2) 磁盘缓存
        vector = self._get_disk_locked(key, now)
        if vector is not None:
            self._put_memory_locked(key, vector, now)
            self._hits += 1
            self._disk_hits += 1
            return list(vector)

        self._misses += 1
        return None

    def _is_expired(self, accessed_at: float, now: float) -> bool:
        """判断缓存条目是否过期；ttl_seconds=0 表示永不过期。"""
        return self.ttl_seconds > 0 and (now - accessed_at) > self.ttl_seconds

    def _put_memory_locked(self, key: str, vector: List[float], now: float) -> None:
        """写入内存 LRU；超过容量时淘汰最久未使用的条目。"""
        if self.max_memory_items <= 0:
            return
        self._memory[key] = (list(vector), now)
        self._memory.move_to_end(key)
        while len(self._memory) > self.max_memory_items:
            self._memory.popitem(last=False)

    # ------------------------------------------------------------------
    # 内部工具：磁盘
    # ------------------------------------------------------------------
    def _ensure_disk_locked(self) -> bool:
        """确保磁盘缓存可用（建目录、建表）；失败时退化为纯内存缓存。"""
        if not self.enabled or not self.db_path or self._disk_broken:
            return False
        if self._disk_ready and self._conn is not None:
            return True
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False：连接会被多个线程共用，外层已用 RLock 串行化
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(self._SCHEMA)
            conn.commit()
            self._conn = conn
            self._disk_ready = True
            return True
        except (OSError, sqlite3.Error) as exc:
            self._errors += 1
            self._disk_broken = True
            logger.warning(
                "向量磁盘缓存不可用，已退化为纯内存缓存（%s）：%s",
                self.db_path,
                exc,
            )
            return False

    def _get_disk_locked(self, key: str, now: float) -> Optional[List[float]]:
        """从 SQLite 读取并解码向量；过期或损坏的记录会被删除/忽略。"""
        if not self._ensure_disk_locked():
            return None
        try:
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT vector, dimension, last_access FROM embeddings WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            blob, dimension, last_access = row
            if self._is_expired(float(last_access), now):
                self._conn.execute(
                    "DELETE FROM embeddings WHERE cache_key = ?", (key,)
                )
                self._conn.commit()
                return None

            vector = array("f")
            vector.frombytes(blob)
            if dimension and len(vector) != int(dimension):
                logger.warning(
                    "缓存向量维度不一致（期望 %s，实际 %s），已忽略",
                    dimension,
                    len(vector),
                )
                return None
            result = [float(value) for value in vector]
            if not result or not all(math.isfinite(value) for value in result):
                return None

            # 更新访问时间与命中计数；写失败不影响读取结果
            self._conn.execute(
                "UPDATE embeddings SET last_access = ?, hits = hits + 1 WHERE cache_key = ?",
                (now, key),
            )
            self._conn.commit()
            return result
        except (sqlite3.Error, ValueError, TypeError) as exc:
            self._errors += 1
            logger.warning("读取向量磁盘缓存失败（key=%s）：%s", key, exc)
            return None

    def _put_disk_locked(self, rows: Iterable[tuple]) -> None:
        """批量 UPSERT 到 SQLite；单条失败不会中断其它记录。"""
        if not self._ensure_disk_locked():
            return
        try:
            assert self._conn is not None
            self._conn.executemany(
                """
                INSERT INTO embeddings
                    (cache_key, model, text_hash, dimension, vector,
                     created_at, last_access)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    vector = excluded.vector,
                    dimension = excluded.dimension,
                    last_access = excluded.last_access
                """,
                list(rows),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            self._errors += 1
            logger.warning("写入向量磁盘缓存失败：%s", exc)

    def _disk_stats_locked(self) -> tuple[int, int]:
        """返回 (磁盘缓存条数, 向量字节数)；磁盘不可用时返回 (0, 0)。"""
        if not self._ensure_disk_locked():
            return 0, 0
        try:
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(LENGTH(vector)), 0) FROM embeddings"
            ).fetchone()
            return int(row[0] or 0), int(row[1] or 0)
        except sqlite3.Error:
            return 0, 0


class CachedEmbeddings(Base):
    """向量模型缓存包装器：让任意 ``Base`` 实现自动获得缓存能力。

    使用方式::

        inner = LocalEmbeddings(...)          # 或 OpenAIEmbeddings(...)
        model = CachedEmbeddings(inner, cache, cache_name="offline:BAAI/bge-small-zh")

    设计要点：
    - ``embed_text`` 走批量逻辑，保证单条与批量行为一致；
    - 只对未命中的文本调用真实模型，命中顺序不变；
    - 真实模型返回条数不一致时抛 ValueError，避免错位写入向量库。
    """

    def __init__(
        self,
        inner: Base,
        cache: EmbeddingCache,
        cache_name: Optional[str] = None,
        batch_size: int = 16,
    ) -> None:
        if inner is None:
            raise ValueError("inner 向量模型不能为空")
        if cache is None:
            raise ValueError("cache 缓存实例不能为空")
        self.inner = inner
        self.cache = cache
        # cache_name 会参与缓存键：建议带上“模式 + 模型 + 维度”，避免串用
        self.cache_name = cache_name or inner.get_model_name()
        self.batch_size = max(1, int(batch_size))
        self._dimension: Optional[int] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Base 接口实现
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        """单条文本向量化（内部转成一次批量调用）。"""
        vectors = self.embed_batch([text])
        return vectors[0] if vectors else []

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量向量化：缓存命中直接返回，未命中才调用底层模型。"""
        if texts is None:
            return []
        # 统一转成字符串：下游 OpenAI / SentenceTransformer 都只接受 str
        normalized = ["" if text is None else str(text) for text in texts]
        if not normalized:
            return []

        cached = self.cache.get_many(self.cache_name, normalized)
        # 防御：缓存里可能残留其它维度的旧向量（例如换过 dimension 参数）
        with self._lock:
            expected = self._dimension
        if expected is not None:
            for index, vector in enumerate(cached):
                if vector is not None and len(vector) != expected:
                    cached[index] = None

        miss_indices = [
            index for index, vector in enumerate(cached) if vector is None
        ]
        if miss_indices:
            miss_texts = [normalized[index] for index in miss_indices]
            # 分批调用，避免一次向在线接口发送过多文本（部分平台限制 10~32 条）
            fresh_vectors: List[List[float]] = []
            for start in range(0, len(miss_texts), self.batch_size):
                batch = miss_texts[start : start + self.batch_size]
                fresh_vectors.extend(self.inner.embed_batch(batch))

            if len(fresh_vectors) != len(miss_texts):
                raise ValueError(
                    "向量模型返回条数与请求不一致："
                    f"请求 {len(miss_texts)} 条，返回 {len(fresh_vectors)} 条"
                )

            for index, vector in zip(miss_indices, fresh_vectors):
                cached[index] = [float(value) for value in vector]

            # 记录维度，供后续命中校验
            if fresh_vectors and fresh_vectors[0]:
                with self._lock:
                    self._dimension = len(fresh_vectors[0])

            # 写入缓存失败不应影响本次请求的返回结果
            self.cache.put_many(self.cache_name, miss_texts, fresh_vectors)

        # 理论上不会有 None（除非底层返回空向量），按空列表兜底
        return [vector if vector is not None else [] for vector in cached]

    def get_dimension(self) -> int:
        """返回底层模型的向量维度。"""
        return self.inner.get_dimension()

    def get_model_name(self) -> str:
        """返回底层模型名称（用于展示与统计）。"""
        return self.inner.get_model_name()

    # ------------------------------------------------------------------
    # 扩展能力
    # ------------------------------------------------------------------
    def cache_stats(self) -> dict:
        """暴露底层缓存统计，方便 API 层直接返回。"""
        return self.cache.stats()


# ----------------------------------------------------------------------
# 进程级默认缓存：所有向量模型共享一个 SQLite 缓存文件
# ----------------------------------------------------------------------
_DEFAULT_CACHE: Optional[EmbeddingCache] = None
_DEFAULT_CACHE_LOCK = threading.Lock()


def get_default_embedding_cache() -> EmbeddingCache:
    """获取进程级默认向量缓存（懒加载、线程安全）。"""
    global _DEFAULT_CACHE
    if _DEFAULT_CACHE is not None:
        return _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is None:
            # 延迟导入配置，避免模块导入顺序问题
            from src.api.config import settings

            db_path = str(Path(settings.EMBEDDING_CACHE_DIR) / "embeddings.sqlite3")
            _DEFAULT_CACHE = EmbeddingCache(
                db_path=db_path,
                max_memory_items=settings.EMBEDDING_CACHE_MAX_ITEMS,
                ttl_seconds=settings.EMBEDDING_CACHE_TTL_SECONDS,
                enabled=settings.EMBEDDING_CACHE_ENABLED,
            )
            logger.info("向量缓存已启用：%s", db_path)
        return _DEFAULT_CACHE


def reset_default_embedding_cache() -> None:
    """关闭并重置默认缓存（测试或清空缓存后调用）。"""
    global _DEFAULT_CACHE
    with _DEFAULT_CACHE_LOCK:
        if _DEFAULT_CACHE is not None:
            _DEFAULT_CACHE.close()
        _DEFAULT_CACHE = None
