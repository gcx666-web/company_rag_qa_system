# -*- coding: utf-8 -*-
"""
document_router.py —— 知识库管理接口。

提供前端“知识库”页面所需的能力：

- 集合（知识库）列表 / 创建 / 删除；
- 集合内文件列表（按 file_name 聚合分块，并合并上传登记表中的大小、指纹、
  上传时间）；
- 删除单个文件的全部分块；
- 集合统计信息。

说明：列表接口只读取向量库元数据，不会触发离线向量模型加载；只有删除/统计
等确实需要的操作才会触碰向量模型。
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from src.api.config import settings
from src.api.desp import evict_document_store, get_document_store
from src.api.models import (
    CollectionCreateRequest,
    CollectionInfo,
    DeleteCollectionResponse,
    DeleteFileResponse,
    FileInfo,
)
from src.api.utils import safe_filename, validate_collection_name
from src.database.chroma_client import (
    delete_collection as drop_chroma_collection,
    list_collection_names,
)
from src.database.document_store import DocumentStore
from src.database.upload_registry import get_upload_registry
from src.embeddings.factory import describe_embedding

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["知识库管理"])


def _collection_exists(name: str) -> bool:
    """判断集合是否真实存在（避免“只读列表”顺手创建空集合）。"""
    return name in list_collection_names(settings.CHROMA_PERSIST_DIR)


def _get_store(name: str) -> DocumentStore:
    """构造（缓存的）文档存储；初始化失败时返回带原因的 503。

    ``ChromaVectorStore`` 在构造阶段就要校验 distance 并打开集合，配置写错
    （例如 ``CHROMA_DISTANCE`` 被行内注释污染）或向量库损坏时都会抛异常。
    原实现让异常直接冒到 FastAPI，前端只能看到纯文本 ``Internal Server Error``，
    用户完全不知道问题在哪；这里统一翻译成 503 + 中文原因，前端可原样展示。

    :param name: 集合名称（同时作为写入/检索的 namespace）。
    :raises HTTPException: 503，附带具体失败原因。
    """
    try:
        return get_document_store(name)
    except Exception as exc:
        logger.exception("初始化知识库 %s 失败", name)
        raise HTTPException(
            status_code=503,
            detail=f"知识库「{name}」初始化失败：{exc}",
        ) from exc


def _compatibility(
    stored_dimension: int | None, model_dimension: int | None
) -> bool | None:
    """判断集合向量维度与当前模型维度是否兼容。

    空集合（stored=None）或模型尚未加载（model=None）时无法判断，返回 None；
    前端只在明确为 False 时提示“需要重建知识库”。
    """
    if stored_dimension is None or model_dimension is None:
        return None
    return int(stored_dimension) == int(model_dimension)


@router.get("/collections", summary="集合列表")
def list_collections() -> dict:
    """列出所有知识库集合及分块/文件统计。

    同时返回集合真实向量维度与当前模型维度：切换在线/离线模型后，旧集合可能
    维度不一致，前端据此提前提示，而不是等问答时报 Chroma 维度错误。
    """
    embedding = describe_embedding()
    model_dimension = embedding.get("dimension")
    result: list[dict[str, Any]] = []
    for name in list_collection_names(settings.CHROMA_PERSIST_DIR):
        try:
            store = get_document_store(name)
            info = store.vector_store.get_collection_info()
            files = store.list_files(namespace=name)
            stored_dimension = store.vector_store.get_vector_dimension()
            result.append(
                {
                    "name": name,
                    "count": int(info.get("count", 0) or 0),
                    "distance": info.get("distance", settings.CHROMA_DISTANCE),
                    "files": len(files),
                    "embedding_model": embedding.get("model") or "",
                    "embedding_dimension": stored_dimension,
                    "model_dimension": model_dimension,
                    "compatible": _compatibility(stored_dimension, model_dimension),
                }
            )
        except Exception as exc:
            # 单个集合损坏不应让整个列表接口失败
            logger.warning("读取集合 %s 信息失败：%s", name, exc)
            result.append(
                {
                    "name": name,
                    "count": 0,
                    "distance": settings.CHROMA_DISTANCE,
                    "files": 0,
                    "embedding_model": embedding.get("model") or "",
                    "embedding_dimension": None,
                    "model_dimension": model_dimension,
                    "compatible": None,
                    "error": str(exc),
                }
            )
    return {"collections": result}


@router.post("/collections", response_model=CollectionInfo, summary="创建集合")
def create_collection(body: CollectionCreateRequest) -> CollectionInfo:
    """创建（或获取）一个知识库集合。"""
    name = validate_collection_name(body.name)
    store = _get_store(name)
    info = store.vector_store.get_collection_info()
    embedding = describe_embedding()
    stored_dimension = store.vector_store.get_vector_dimension()
    model_dimension = embedding.get("dimension")
    return CollectionInfo(
        name=name,
        count=int(info.get("count", 0) or 0),
        distance=str(info.get("distance", settings.CHROMA_DISTANCE)),
        files=len(store.list_files(namespace=name)),
        embedding_model=store.embedding_model.get_model_name(),
        embedding_dimension=stored_dimension,
        model_dimension=model_dimension,
        compatible=_compatibility(stored_dimension, model_dimension),
    )


@router.delete(
    "/collections/{collection_name}",
    response_model=DeleteCollectionResponse,
    summary="删除集合",
)
def delete_collection(collection_name: str) -> DeleteCollectionResponse:
    """删除整个集合（不重建空集合，同时清理缓存与上传登记表）。"""
    name = validate_collection_name(collection_name)
    if not _collection_exists(name):
        raise HTTPException(status_code=404, detail=f"集合不存在：{name}")

    # 直接让 Chroma 删除集合，而不是调用 DocumentStore.clear()：
    # clear() 的设计语义是“清空并重建”，用于重置集合内容；
    # 删除知识库则不应该留下一个同名空集合。
    deleted = drop_chroma_collection(name, settings.CHROMA_PERSIST_DIR)
    if deleted:
        evict_document_store(name)
        get_upload_registry().delete_by_collection(name)
    logger.info("集合已删除：%s（结果=%s）", name, deleted)
    return DeleteCollectionResponse(deleted=bool(deleted), collection_name=name)


@router.get("/files", summary="文件列表")
def list_files(
    collection_name: str = Query(..., description="集合名称"),
) -> dict:
    """列出集合内文件，并合并上传登记表中的元数据。"""
    name = validate_collection_name(collection_name)
    if not _collection_exists(name):
        return {"collection_name": name, "files": []}

    store = _get_store(name)
    files = store.list_files(namespace=name)
    records = {
        record.file_name: record
        for record in get_upload_registry().list_by_collection(name)
    }
    merged: list[dict[str, Any]] = []
    for item in files:
        record = records.get(item["file_name"])
        merged.append(
            FileInfo(
                file_name=item["file_name"],
                namespace=item.get("namespace", ""),
                chunks=int(item.get("chunks", 0) or 0),
                tables=int(item.get("tables", 0) or 0),
                pages=list(item.get("pages") or []),
                total_chars=int(item.get("total_chars", 0) or 0),
                size=record.size if record else None,
                sha256=record.sha256 if record else None,
                uploaded_at=record.updated_at if record else None,
                status=record.status if record else "success",
                error=record.error if record else "",
            ).model_dump()
        )
    return {"collection_name": name, "files": merged}


@router.delete("/files", response_model=DeleteFileResponse, summary="删除文件")
def delete_file(
    collection_name: str = Query(..., description="集合名称"),
    file_name: str = Query(..., description="文件名"),
) -> DeleteFileResponse:
    """删除某个文件在集合中的全部分块，并清理登记记录。"""
    name = validate_collection_name(collection_name)
    safe_name = safe_filename(file_name)
    store = _get_store(name)
    removed = store.delete_by_file_name(safe_name, namespace=name)
    get_upload_registry().delete_by_name(name, safe_name)
    logger.info("删除文件分块：collection=%s, file=%s, removed=%d", name, safe_name, removed)
    return DeleteFileResponse(
        deleted=removed > 0,
        file_name=safe_name,
        collection_name=name,
        removed_chunks=int(removed or 0),
    )


@router.get("/stats", summary="集合统计")
def collection_stats(
    collection_name: str = Query(..., description="集合名称"),
) -> dict:
    """返回集合的向量数、文件数、向量模型与缓存统计。"""
    name = validate_collection_name(collection_name)
    store = _get_store(name)
    stats = store.get_stats()
    stats["files"] = len(store.list_files(namespace=name))
    stats["registry_records"] = get_upload_registry().count(name)
    return stats
