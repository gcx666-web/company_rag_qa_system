#上传文件路由
from fastapi import APIRouter, UploadFile, File

from src.api.config import settings
from src.parsers import ParserFactory, parse_file
router = APIRouter( prefix="/document", tags=["上传文件"])

@router.post("/upload/", summary="上传文件")
async def upload_file(file: UploadFile = File(...)):
    """
    上传并解析文件
    支持的文件类型: PDF, DOCX, HTML, MD
    """
    #判断文件名为空
    if not file.filename:
        raise ValueError("文件名不能为空")
    #判断文件后缀
    if not str(file.filename).endswith(tuple(ParserFactory.support_file_types())):
        raise ValueError("不支持的文件格式")
    #判断文件大小
    contents = await file.read()
    if len(contents) > settings.MAX_FILE_SIZE:
        raise ValueError("文件大小超过限制")
    #保存文件
    _ensure_upload_dir_exists()
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)
    try:
        #获取解析器
        parser = ParserFactory.get_parser(file.filename)
        result = parser.parse_from_bytes(contents, file.filename)
        return {
            "message":"文件处理成功",
            "file_name":file.filename,
            "total_pages":result.total_pages,
            "chunks_count":result.total_chunks,
            "table_count":result.table_count,
            "chunks":result.chunks,
        }
    except Exception as e:
        return {
            "message":"文件处理失败",
            "error":str(e)
        }
from pydantic import BaseModel
class ParsePathRequest(BaseModel):
    file_path: str
from fastapi import HTTPException
import os

#确保上传目录存在
def _ensure_upload_dir_exists():
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

@router.post("/parse-path", summary="解析已有文件")
async def parse_document_by_path(body:ParsePathRequest):
    """
    解析已有文件
    支持的文件类型: PDF, DOCX, HTML, MD
    """
    #判断文件名为空
    if not body.file_path:
        raise HTTPException(status_code=400, detail="文件路径不能为空")
    #判断文件是否存在
    if not os.path.exists(body.file_path):
        raise HTTPException(status_code=400, detail="文件不存在")
    #判断是否完整路径
    if not os.path.isabs(body.file_path):
        raise HTTPException(status_code=400, detail="请提供完整的文件路径")
    #判断文件后缀
    if not str(body.file_path).endswith(tuple(ParserFactory.support_file_types())):
        raise HTTPException(status_code=400, detail="不支持的文件格式")
    #判断文件大小
    if os.path.getsize(body.file_path) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="文件大小超过限制")
    try:
        result = parse_file(body.file_path)
        return {
            "message":"文件处理成功",
            "file_name":os.path.basename(body.file_path),
            "total_pages":result.total_pages,
            "chunks_count":result.total_chunks,
            "table_count":result.table_count,
            "chunks":result.chunks,
        }
    except Exception as e:
        return {
            "message":"文件处理失败",
            "error":str(e)
        }
@router.post("/support-types", summary="支持类型")
async def get_support_file_types():
    """
    获取支持的文件类型
    """
    return {
        "support_file_types": ParserFactory.support_file_types()
    }
    