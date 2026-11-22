# -*- coding: utf-8 -*-
"""
upload_registry.py —— 上传登记表 + 并发上传锁。

要解决的问题
------------
用户重复上传同一个文件时，原实现会：

1. 重新解析整份文档（Docling 对扫描版 PDF 可能耗时数分钟）；
2. 重新计算全部分块向量（在线接口要花钱，离线模型要占 CPU）；
3. 用确定性 id 写入向量库，触发 DuplicateIDError 或覆盖逻辑。

本模块用“内容指纹 + 登记表”解决：

- 上传时计算文件 SHA-256；
- 登记表以 ``(collection, file_name)`` 为唯一键，记录 sha256、大小、分块数、
  上传时间、状态；
- 相同集合 + 相同 sha256 再次上传，直接返回登记信息，跳过解析/向量化（秒传）；
- 同名文件但内容不同，按“覆盖导入”处理，并更新登记表；
- 同名文件并发上传用进程内锁串行化，避免两个请求同时删除/写入同一批 id。

登记表使用 SQLite（WAL 模式），进程重启后仍然有效，多线程访问由锁串行化。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "UploadRecord",
    "UploadRegistry",
    "UploadLockPool",
    "get_upload_registry",
    "get_upload_lock_pool",
]


@dataclass
class UploadRecord:
    """一条上传登记记录。"""

    collection: str
    file_name: str
    sha256: str
    size: int = 0
    pages: int = 0
    chunks: int = 0
    tables: int = 0
    # success / processing / failed
    status: str = "success"
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """转成可 JSON 序列化的字典。"""
        return asdict(self)


class UploadRegistry:
    """上传登记表（SQLite）。

    :param db_path: SQLite 文件路径。
    :param enabled: 总开关；关闭后所有查询返回 None、写入直接跳过。
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS uploads (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        collection TEXT NOT NULL,
        file_name  TEXT NOT NULL,
        sha256     TEXT NOT NULL,
        size       INTEGER NOT NULL DEFAULT 0,
        pages      INTEGER NOT NULL DEFAULT 0,
        chunks     INTEGER NOT NULL DEFAULT 0,
        tables     INTEGER NOT NULL DEFAULT 0,
        status     TEXT NOT NULL DEFAULT 'success',
        error      TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        UNIQUE(collection, file_name)
    );
    CREATE INDEX IF NOT EXISTS idx_uploads_hash
        ON uploads(collection, sha256);
    """

    def __init__(self, db_path: str, enabled: bool = True) -> None:
        self.db_path = str(db_path)
        self.enabled = bool(enabled)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ready = False
        self._broken = False

    # ------------------------------------------------------------------
    # 连接 / 建表
    # ------------------------------------------------------------------
    def _ensure_ready(self) -> bool:
        """确保数据库可用；失败时退化为“无登记表”模式，不影响上传主流程。"""
        if not self.enabled or self._broken:
            return False
        if self._ready and self._conn is not None:
            return True
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.executescript(self._SCHEMA)
            conn.commit()
            self._conn = conn
            self._ready = True
            return True
        except (OSError, sqlite3.Error) as exc:
            self._broken = True
            logger.warning("上传登记表不可用（%s）：%s", self.db_path, exc)
            return False

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_by_name(
        self, collection: str, file_name: str
    ) -> Optional[UploadRecord]:
        """按 (集合, 文件名) 查询登记记录。"""
        if not self._ensure_ready():
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM uploads WHERE collection = ? AND file_name = ?",
                    (collection, file_name),
                ).fetchone()
            except sqlite3.Error as exc:
                logger.warning("查询上传登记失败：%s", exc)
                return None
        return self._row_to_record(row) if row else None

    def get_by_hash(
        self, collection: str, sha256: str
    ) -> Optional[UploadRecord]:
        """按 (集合, 内容指纹) 查询最近一条登记记录。"""
        if not self._ensure_ready():
            return None
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT * FROM uploads
                    WHERE collection = ? AND sha256 = ?
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (collection, sha256),
                ).fetchone()
            except sqlite3.Error as exc:
                logger.warning("按内容指纹查询上传登记失败：%s", exc)
                return None
        return self._row_to_record(row) if row else None

    def list_by_collection(self, collection: str) -> list[UploadRecord]:
        """列出一个集合下的所有上传记录。"""
        if not self._ensure_ready():
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM uploads WHERE collection = ? ORDER BY updated_at DESC",
                    (collection,),
                ).fetchall()
            except sqlite3.Error as exc:
                logger.warning("列出上传登记失败：%s", exc)
                return []
        return [self._row_to_record(row) for row in rows]

    def count(self, collection: Optional[str] = None) -> int:
        """统计登记记录数量。"""
        if not self._ensure_ready():
            return 0
        with self._lock:
            try:
                if collection is None:
                    row = self._conn.execute("SELECT COUNT(*) FROM uploads").fetchone()
                else:
                    row = self._conn.execute(
                        "SELECT COUNT(*) FROM uploads WHERE collection = ?",
                        (collection,),
                    ).fetchone()
                return int(row[0] or 0)
            except sqlite3.Error:
                return 0

    # ------------------------------------------------------------------
    # 写入 / 更新 / 删除
    # ------------------------------------------------------------------
    def upsert(self, record: UploadRecord) -> bool:
        """写入或更新登记记录（以 collection + file_name 为唯一键）。"""
        if not self._ensure_ready():
            return False
        now = time.time()
        if not record.created_at:
            record.created_at = now
        record.updated_at = now
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO uploads
                        (collection, file_name, sha256, size, pages, chunks,
                         tables, status, error, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(collection, file_name) DO UPDATE SET
                        sha256 = excluded.sha256,
                        size = excluded.size,
                        pages = excluded.pages,
                        chunks = excluded.chunks,
                        tables = excluded.tables,
                        status = excluded.status,
                        error = excluded.error,
                        updated_at = excluded.updated_at
                    """,
                    (
                        record.collection,
                        record.file_name,
                        record.sha256,
                        int(record.size),
                        int(record.pages),
                        int(record.chunks),
                        int(record.tables),
                        record.status,
                        record.error,
                        float(record.created_at),
                        float(record.updated_at),
                    ),
                )
                self._conn.commit()
                return True
            except sqlite3.Error as exc:
                logger.warning("写入上传登记失败：%s", exc)
                return False

    def mark_processing(
        self, collection: str, file_name: str, sha256: str, size: int
    ) -> None:
        """标记“处理中”，便于失败后排查与前端展示历史状态。"""
        self.upsert(
            UploadRecord(
                collection=collection,
                file_name=file_name,
                sha256=sha256,
                size=size,
                status="processing",
            )
        )

    def mark_failed(
        self,
        collection: str,
        file_name: str,
        sha256: str,
        size: int,
        error: str,
    ) -> None:
        """标记失败并保留错误摘要，不删除历史统计。"""
        self.upsert(
            UploadRecord(
                collection=collection,
                file_name=file_name,
                sha256=sha256,
                size=size,
                status="failed",
                error=str(error)[:1000],
            )
        )

    def delete_by_name(self, collection: str, file_name: str) -> bool:
        """删除一条登记记录。"""
        if not self._ensure_ready():
            return False
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM uploads WHERE collection = ? AND file_name = ?",
                    (collection, file_name),
                )
                self._conn.commit()
                return True
            except sqlite3.Error as exc:
                logger.warning("删除上传登记失败：%s", exc)
                return False

    def delete_by_collection(self, collection: str) -> bool:
        """删除一个集合下的所有登记记录（删除集合时调用）。"""
        if not self._ensure_ready():
            return False
        with self._lock:
            try:
                self._conn.execute(
                    "DELETE FROM uploads WHERE collection = ?", (collection,)
                )
                self._conn.commit()
                return True
            except sqlite3.Error as exc:
                logger.warning("删除集合上传登记失败：%s", exc)
                return False

    def close(self) -> None:
        """关闭数据库连接。"""
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except sqlite3.Error:
                    pass
                self._conn = None
                self._ready = False

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> UploadRecord:
        """把 SQLite 行转成 dataclass。"""
        return UploadRecord(
            collection=str(row["collection"]),
            file_name=str(row["file_name"]),
            sha256=str(row["sha256"]),
            size=int(row["size"] or 0),
            pages=int(row["pages"] or 0),
            chunks=int(row["chunks"] or 0),
            tables=int(row["tables"] or 0),
            status=str(row["status"] or "success"),
            error=str(row["error"] or ""),
            created_at=float(row["created_at"] or 0.0),
            updated_at=float(row["updated_at"] or 0.0),
        )


class UploadLockPool:
    """进程内上传锁池：按 key 串行化同一文件/同一内容的并发上传。

    同一内容（collection + sha256）和同一文件名（collection + file_name）
    会分别取锁；排序后依次获取，避免两个请求以相反顺序拿锁造成死锁。
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}
        self._refs: dict[str, int] = {}

    @contextmanager
    def hold(self, *keys: str) -> Iterator[None]:
        """获取一组锁；退出时释放并清理无人使用的锁对象。"""
        normalized = sorted({str(key) for key in keys if key})
        acquired: list[tuple[str, threading.Lock]] = []

        with self._guard:
            for key in normalized:
                lock = self._locks.get(key)
                if lock is None:
                    lock = threading.Lock()
                    self._locks[key] = lock
                self._refs[key] = self._refs.get(key, 0) + 1
                acquired.append((key, lock))

        try:
            # 严格按排序后的顺序加锁，避免交叉等待
            for _, lock in acquired:
                lock.acquire()
            yield
        finally:
            for _, lock in acquired:
                lock.release()
            with self._guard:
                for key, _ in acquired:
                    remaining = self._refs.get(key, 1) - 1
                    if remaining <= 0:
                        self._refs.pop(key, None)
                        self._locks.pop(key, None)
                    else:
                        self._refs[key] = remaining


# ----------------------------------------------------------------------
# 进程级单例
# ----------------------------------------------------------------------
_REGISTRY: Optional[UploadRegistry] = None
_REGISTRY_LOCK = threading.Lock()
_LOCK_POOL = UploadLockPool()


def get_upload_registry() -> UploadRegistry:
    """获取进程级上传登记表（懒加载）。"""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            from src.api.config import settings

            _REGISTRY = UploadRegistry(
                db_path=settings.UPLOAD_REGISTRY_DB,
                enabled=settings.UPLOAD_DEDUP_ENABLED,
            )
            logger.info(
                "上传登记表已启用：%s（去重=%s）",
                settings.UPLOAD_REGISTRY_DB,
                settings.UPLOAD_DEDUP_ENABLED,
            )
    return _REGISTRY


def get_upload_lock_pool() -> UploadLockPool:
    """获取进程级上传锁池。"""
    return _LOCK_POOL
