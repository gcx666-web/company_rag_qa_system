#上传文件路由
import logging
import os
from pathlib import Path

from chromadb.errors import DuplicateIDError
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.api.config import settings
from src.api.desp import get_document_store
from src.parsers import ParserFactory, parse_file

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter( prefix="/document", tags=["上传文件"])

#Chroma 集合名约束：3-512 个字符，只允许 [a-zA-Z0-9._-]，且首尾必须是字母或数字
_COLLECTION_NAME_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


class ParsePathRequest(BaseModel):
    '''解析已有文件的请求体'''
    file_path: str


def _validate_collection_name(name) -> str:
    '''
    校验 Chroma 集合名，非法时直接返回 400

    下游 Chroma 对不合法的集合名会抛 InvalidArgumentError，
    如果不在入口拦下，就会被包装成笼统的“文件处理失败”。
    '''
    name = str(name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="集合名称不能为空")
    if not 3 <= len(name) <= 512:
        raise HTTPException(
            status_code=400, detail="集合名称长度必须在 3-512 个字符之间"
        )
    if set(name) - _COLLECTION_NAME_CHARS:
        raise HTTPException(
            status_code=400, detail="集合名称只能包含字母、数字、下划线、点和短横线"
        )
    if not (name[0].isalnum() and name[-1].isalnum()):
        raise HTTPException(
            status_code=400, detail="集合名称必须以字母或数字开头和结尾"
        )
    return name


def _ensure_upload_dir_exists():
    '''确保上传目录存在'''
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


def _validate_upload(filename: str, size: int) -> None:
    '''
    校验上传文件名、扩展名与大小，非法时直接返回 400 / 413
    '''
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    #扩展名统一转小写再比对，.PDF / .HTM 这类大写后缀也能正常放行
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


def _parse_and_store(contents: bytes, filename: str, collection_name) -> dict:
    '''
    解析文件；collection_name 有值时写入向量库（同名文件覆盖旧分块）

    Docling 解析与向量化都是同步阻塞的重活（大 PDF 可达分钟级），
    所以由 run_in_threadpool 在子线程执行，避免阻塞事件循环。
    '''
    #获取解析器
    parser = ParserFactory.get_parser(filename)
    result = parser.parse_from_bytes(contents, filename)
    #是否保存到数据库
    if collection_name is None:
        return {
            "message": "文件处理成功",
            "file_name": filename,
            "total_pages": result.total_pages,
            "chunks_count": result.total_chunks,
            "table_count": result.table_count,
            "chunks": result.chunks,
        }

    #获取文档存储服务（密钥/向量库/持久化目录统一由 desp 解析）
    document_store = get_document_store(collection_name)
    #overwrite=True：重新上传同一个文件时先删旧分块再写新分块（方案 A：覆盖导入）
    ids = document_store.add_parse_result(
        result, namespace=collection_name, overwrite=True
    )
    return {
        "message": "文件保存成功",
        "file_name": filename,
        "collection_name": collection_name,
        "total_pages": result.total_pages,
        "chunks_count": result.total_chunks,
        "table_count": result.table_count,
        "ids": ids,
    }


@router.post("/upload/", summary="上传文件")
async def upload_file(file: UploadFile = File(...), collection_name: str | None = None):
    """
    上传并解析文件，并判断是否保存到数据库
    支持的文件类型: PDF, DOCX, HTML, MD
    参数:
    file: 上传的文件
    collection_name: 集合名称，如果有，则保存到数据库（同名文件覆盖旧分块），没有则不保存
    """
    #保存到磁盘用去掉目录的文件名，避免 ../ 之类的路径穿越
    filename = os.path.basename(str(file.filename or ""))
    contents = await file.read()
    await file.close()
    #校验文件名/后缀/大小（返回值即 400 / 413）
    _validate_upload(filename, len(contents))
    #校验集合名，不合法时直接 400
    collection = _validate_collection_name(collection_name) if collection_name else None

    #保存文件
    _ensure_upload_dir_exists()
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        return await run_in_threadpool(
            _parse_and_store, contents, filename, collection
        )
    except HTTPException:
        raise
    except DuplicateIDError as exc:
        #overwrite=True 会先删旧分块，正常不会走到这里，兜底给出可读提示
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        #记录完整堆栈，便于排查“处理失败”的真实原因
        logger.exception(
            "文件处理失败：file=%s, collection=%s", filename, collection
        )
        raise HTTPException(status_code=500, detail=f"文件处理失败: {exc}") from exc


@router.post("/parse-path", summary="解析已有文件")
def parse_document_by_path(body: ParsePathRequest):
    """
    解析已有文件
    支持的文件类型: PDF, DOCX, HTML, MD
    """
    file_path = str(body.file_path or "").strip()
    #判断文件路径为空
    if not file_path:
        raise HTTPException(status_code=400, detail="文件路径不能为空")
    #先判断是否绝对路径，再判断存在性（否则相对路径会被误报“文件不存在”）
    if not os.path.isabs(file_path):
        raise HTTPException(status_code=400, detail="请提供完整的文件路径")
    #判断文件是否存在
    if not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="文件不存在")
    #判断文件后缀（大小写不敏感）
    ext = Path(file_path).suffix.lower()
    if ext not in ParserFactory.support_file_types():
        raise HTTPException(
            status_code=400, detail=f"不支持的文件格式: {ext or '未知'}"
        )
    #判断文件大小
    if os.path.getsize(file_path) > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"文件大小超过限制（最大 {max_mb} MB）"
        )

    try:
        #同步路由由 FastAPI 自动放进线程池执行，不会阻塞事件循环
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


@router.post("/support-types", summary="支持类型")
async def get_support_file_types():
    """
    获取支持的文件类型
    """
    return {
        "support_file_types": ParserFactory.support_file_types()
    }