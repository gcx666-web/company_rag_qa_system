# -*- coding: utf-8 -*-
"""
upload_file_router.py —— 文件上传 / 解析接口。

相比原实现，这里加入了三层优化：

1. **内容去重（秒传）**：上传时计算 SHA-256，命中上传登记表且向量库确认存在
   该文件分块时，直接返回登记信息，不再解析、不再向量化；
2. **覆盖导入**：同名文件内容变化时自动删除旧分块后写入新分块，并用登记表
   标记为 ``replaced``；
3. **缓存命中统计**：响应里返回本次请求期间新增的向量缓存命中次数，重复上传
   或强制重导时能直观看到“没有重复调用向量模型”。

并发控制：同一内容 / 同一文件名的上传通过进程内锁串行化，避免两个请求同时
删除、写入同一批确定性 id。
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

from chromadb.errors import DuplicateIDError
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from src.api.config import settings
from src.api.desp import get_document_store
from src.api.models import ParsePathRequest, UploadResponse
from src.api.utils import (
    atomic_write_bytes,
    embedding_dimension_hint,
    safe_filename,
    sha256_bytes,
    validate_collection_name,
)
from src.database.upload_registry import (
    UploadRecord,
    get_upload_lock_pool,
    get_upload_registry,
)
from src.embeddings.embedding_cache import get_default_embedding_cache
from src.parsers import ParserFactory, parse_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["上传文件"])


# ======================================================================
# 一、校验与工具
# ======================================================================
def _ensure_upload_dir_exists() -> None:
    """确保上传目录存在。"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


def _validate_upload(filename: str, size: int) -> None:
    """
    校验上传文件名、扩展名与大小，非法时直接返回 400 / 413。
    """
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    # 扩展名统一转小写再比对，.PDF / .HTM 这类大写后缀也能正常放行
    ext = Path(filename).suffix.lower()
    if ext not in ParserFactory.support_file_types():
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {ext or '未知'}，支持 {ParserFactory.support_file_types()}",
        )
    if size > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"文件大小超过限制（最大 {max_mb} MB）"
        )


def _cache_hits() -> int:
    """读取向量缓存累计命中次数；缓存不可用时返回 0。"""
    try:
        return int(get_default_embedding_cache().stats().get("hits", 0))
    except Exception:
        return 0


def _save_upload(
    contents: bytes, filename: str, collection_name: Optional[str]
) -> Path:
    """把上传文件原子写入磁盘。

    有集合时按 ``uploads/<集合>/<文件名>`` 归档；只解析不入库时放到
    ``uploads/_parsed``，避免和正式知识库文件混在一起。
    """
    _ensure_upload_dir_exists()
    folder = Path(settings.UPLOAD_DIR) / (collection_name or "_parsed")
    target = folder / filename
    return atomic_write_bytes(target, contents)


def _parse_only(contents: bytes, filename: str):
    """只解析、不写向量库（用于预览 / 仅解析接口）。"""
    parser = ParserFactory.get_parser(filename)
    return parser.parse_from_bytes(contents, filename)


def _parse_and_store(
    contents: bytes, filename: str, collection_name: str
) -> Tuple[object, list[str]]:
    """
    解析文件并写入向量库（overwrite=True：同名文件先删旧分块再写新分块）。

    Docling 解析与向量化都是同步阻塞重活，由 run_in_threadpool 放到子线程执行，
    避免阻塞 FastAPI 事件循环。
    """
    parser = ParserFactory.get_parser(filename)
    result = parser.parse_from_bytes(contents, filename)
    document_store = get_document_store(collection_name)
    ids = document_store.add_parse_result(
        result, namespace=collection_name, overwrite=True
    )
    return result, ids


def _duplicate_response(
    record: UploadRecord,
    filename: str,
    collection: str,
    digest: str,
    started: float,
    existing_file_name: Optional[str] = None,
) -> UploadResponse:
    """构造“秒传”响应：不解析、不向量化，直接返回上次的统计信息。"""
    file_name = existing_file_name or record.file_name
    if file_name == filename:
        message = "文件内容已存在，已跳过解析与向量化（秒传）"
    else:
        message = (
            f"相同内容已作为「{file_name}」导入过，已跳过本次解析与向量化（秒传）"
        )
    return UploadResponse(
        status="duplicate",
        message=message,
        file_name=filename,
        collection_name=collection,
        sha256=digest,
        duplicate=True,
        replaced=False,
        total_pages=record.pages,
        chunks_count=record.chunks,
        table_count=record.tables,
        ids=[],
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        embedding_cache_hits=0,
        existing_file_name=file_name if file_name != filename else None,
    )


# ======================================================================
# 二、上传接口
# ======================================================================
@router.post("/upload/", response_model=UploadResponse, summary="上传文件")
async def upload_file(
    file: UploadFile = File(...),
    collection_name: str | None = None,
    force: bool = Query(
        False, description="忽略去重检查，强制重新解析并导入（覆盖同名文件）"
    ),
):
    """
    上传并解析文件；传入 collection_name 时写入向量库。

    - 支持的文件类型：PDF、DOCX、HTML、MD；
    - 相同内容重复上传：默认秒传，设置 ``force=true`` 可强制重导；
    - 同名文件内容变化：自动覆盖旧分块；
    - 返回 ``status``：created / replaced / duplicate。
    """
    started = time.perf_counter()

    # 1) 读取 + 基础校验
    filename = safe_filename(file.filename)
    contents = await file.read()
    await file.close()
    _validate_upload(filename, len(contents))
    collection = (
        validate_collection_name(collection_name) if collection_name else None
    )
    digest = sha256_bytes(contents)
    hits_before = _cache_hits()

    # 2) 不指定集合：只解析并返回分块，不写向量库
    if collection is None:
        saved_path = _save_upload(contents, filename, None)
        try:
            result = await run_in_threadpool(_parse_only, contents, filename)
        except Exception as exc:
            logger.exception("文件解析失败：file=%s, path=%s", filename, saved_path)
            raise HTTPException(
                status_code=500, detail=f"文件处理失败: {exc}"
            ) from exc
        return UploadResponse(
            status="created",
            message="文件处理成功（未指定集合，未写入向量库）",
            file_name=filename,
            collection_name=None,
            sha256=digest,
            duplicate=False,
            replaced=False,
            total_pages=result.total_pages,
            chunks_count=result.total_chunks,
            table_count=result.table_count,
            ids=[],
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            embedding_cache_hits=0,
            chunks=result.chunks,
        )

    # 3) 指定集合：去重 + 覆盖导入
    registry = get_upload_registry()
    lock_pool = get_upload_lock_pool()
    # 同时对“内容”和“文件名”加锁：防止同内容换名并发、同名换内容并发
    with lock_pool.hold(
        f"{collection}:sha:{digest}", f"{collection}:name:{filename}"
    ):
        existing_by_hash = (
            registry.get_by_hash(collection, digest)
            if settings.UPLOAD_DEDUP_ENABLED
            else None
        )

        # 3.1 相同内容已成功导入：确认向量库里确实还有分块，再秒传
        if existing_by_hash is not None and not force:
            if existing_by_hash.status == "processing":
                raise HTTPException(
                    status_code=409,
                    detail="相同文件正在处理中，请稍后刷新知识库",
                )
            if existing_by_hash.status == "success":
                document_store = get_document_store(collection)
                same_name = existing_by_hash.file_name == filename
                if same_name:
                    still_exists = document_store.has_file(
                        filename, namespace=collection
                    )
                    if still_exists:
                        return _duplicate_response(
                            existing_by_hash,
                            filename,
                            collection,
                            digest,
                            started,
                        )
                else:
                    # 相同内容、不同文件名：默认也秒传，避免重复占用向量库
                    still_exists = document_store.has_file(
                        existing_by_hash.file_name, namespace=collection
                    )
                    if still_exists:
                        return _duplicate_response(
                            existing_by_hash,
                            filename,
                            collection,
                            digest,
                            started,
                            existing_file_name=existing_by_hash.file_name,
                        )

        # 3.2 判断是否覆盖旧版本：同名文件但内容指纹不同
        document_store = get_document_store(collection)
        existing_by_name = registry.get_by_name(collection, filename)
        replaced = bool(
            existing_by_name is not None
            and existing_by_name.sha256 != digest
            and existing_by_name.status == "success"
        )
        # 登记表可能缺失（例如手工删过库），再向向量库确认一次
        if not replaced and document_store.has_file(filename, namespace=collection):
            replaced = True

        # 3.3 先登记“处理中”，再落盘、解析、写库
        registry.mark_processing(collection, filename, digest, len(contents))
        saved_path = _save_upload(contents, filename, collection)
        logger.info(
            "开始处理上传：collection=%s, file=%s, sha=%s, replaced=%s, force=%s",
            collection,
            filename,
            digest[:12],
            replaced,
            force,
        )

        try:
            result, ids = await run_in_threadpool(
                _parse_and_store, contents, filename, collection
            )
        except HTTPException:
            registry.mark_failed(collection, filename, digest, len(contents), "HTTPException")
            raise
        except DuplicateIDError as exc:
            # overwrite=True 会先删旧分块，正常不会走到这里，兜底给出可读提示
            registry.mark_failed(
                collection, filename, digest, len(contents), str(exc)
            )
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            # 向量维度不一致时给出可操作的中文提示（409），不要把英文堆栈丢给前端
            hint = embedding_dimension_hint(exc, collection)
            registry.mark_failed(
                collection, filename, digest, len(contents), hint or str(exc)
            )
            logger.exception(
                "文件处理失败：file=%s, collection=%s, path=%s",
                filename,
                collection,
                saved_path,
            )
            if hint:
                raise HTTPException(status_code=409, detail=hint) from exc
            raise HTTPException(
                status_code=500, detail=f"文件处理失败: {exc}"
            ) from exc

        # 3.4 处理成功：更新登记表，返回统计与缓存命中增量
        registry.upsert(
            UploadRecord(
                collection=collection,
                file_name=filename,
                sha256=digest,
                size=len(contents),
                pages=int(getattr(result, "total_pages", 0) or 0),
                chunks=int(getattr(result, "total_chunks", 0) or 0),
                tables=int(getattr(result, "table_count", 0) or 0),
                status="success",
            )
        )
        hits_after = _cache_hits()
        return UploadResponse(
            status="replaced" if replaced else "created",
            message=(
                "同名文件已更新为新版本（旧分块已删除）"
                if replaced
                else "文件保存成功"
            ),
            file_name=filename,
            collection_name=collection,
            sha256=digest,
            duplicate=False,
            replaced=replaced,
            total_pages=int(getattr(result, "total_pages", 0) or 0),
            chunks_count=int(getattr(result, "total_chunks", 0) or 0),
            table_count=int(getattr(result, "table_count", 0) or 0),
            ids=list(ids or []),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            # 本次请求新增的缓存命中（强制重导时命中越多，说明省下的向量计算越多）
            embedding_cache_hits=max(0, hits_after - hits_before),
        )


# ======================================================================
# 三、解析已有文件 / 支持类型
# ======================================================================
@router.post("/parse-path", summary="解析已有文件")
def parse_document_by_path(body: ParsePathRequest):
    """
    解析服务器本地已有文件（不写入向量库）。

    支持的文件类型：PDF、DOCX、HTML、MD。
    """
    file_path = str(body.file_path or "").strip()
    if not file_path:
        raise HTTPException(status_code=400, detail="文件路径不能为空")
    # 先判断是否绝对路径，再判断存在性（否则相对路径会被误报“文件不存在”）
    if not os.path.isabs(file_path):
        raise HTTPException(status_code=400, detail="请提供完整的文件路径")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="文件不存在")
    ext = Path(file_path).suffix.lower()
    if ext not in ParserFactory.support_file_types():
        raise HTTPException(
            status_code=400, detail=f"不支持的文件格式: {ext or '未知'}"
        )
    if os.path.getsize(file_path) > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"文件大小超过限制（最大 {max_mb} MB）"
        )

    try:
        # 同步路由由 FastAPI 自动放进线程池执行，不会阻塞事件循环
        result = parse_file(file_path)
    except Exception as exc:
        logger.exception("文件处理失败：path=%s", file_path)
        raise HTTPException(status_code=500, detail=f"文件处理失败: {exc}") from exc

    return {
        "message": "文件处理成功",
        "file_name": os.path.basename(file_path),
        "total_pages": result.total_pages,
        "chunks_count": result.total_chunks,
        "table_count": result.table_count,
        "chunks": result.chunks,
    }


@router.get("/support-types", summary="支持类型")
@router.post("/support-types", summary="支持类型")
async def get_support_file_types():
    """获取支持的文件类型。"""
    return {"support_file_types": ParserFactory.support_file_types()}
