# -*- coding: utf-8 -*-
"""API 请求/响应模型。

统一放在这里，方便前端对照 OpenAPI 文档（/docs）联调。
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from src.api.config import settings


# ======================================================================
# 对话
# ======================================================================
class ChatRequest(BaseModel):
    """问答请求模型。"""

    question: str = Field(..., min_length=1, description="问题")
    max_tokens: int = Field(
        default=settings.LLM_MAX_TOKENS,
        gt=0,
        description="最大生成长度",
    )
    model: str = Field(
        default=settings.Default_LLM_MODEL or "",
        description="模型名称或供应商名称（zhipu/qwen/siliconflow）",
    )
    collection_name: str = Field(
        default="default",
        description="向量库（集合）名称",
    )
    top_k: int = Field(
        default=settings.TOP_K,
        ge=1,
        le=50,
        description="检索返回的分块数量",
    )


class ChatResponse(BaseModel):
    """问答响应模型。"""

    answer: str = Field(..., description="答案")
    question: str = Field(..., description="问题")
    context: str = Field(default="", description="拼装后的上下文")
    tokens_used: Any = Field(default_factory=dict, description="本次调用的大模型 token 用量")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="来源")
    total_time_ms: float = Field(default=0.0, description="总耗时（毫秒）")
    retrieval_time_ms: float = Field(default=0.0, description="检索耗时（毫秒）")
    model: str = Field(default="", description="实际使用的 LLM 模型名")
    collection_name: str = Field(default="default", description="使用的集合名")


# ======================================================================
# 文档上传
# ======================================================================
class UploadResponse(BaseModel):
    """上传/解析响应。

    ``status`` 的语义：

    - ``created``：新文件首次导入；
    - ``replaced``：同名文件内容变化，已删除旧分块并写入新内容；
    - ``duplicate``：相同内容已存在，本次直接“秒传”，没有重新解析和向量化。
    """

    status: Literal["created", "replaced", "duplicate"] = "created"
    message: str = "文件保存成功"
    file_name: str = ""
    collection_name: Optional[str] = None
    sha256: str = ""
    duplicate: bool = False
    replaced: bool = False
    total_pages: int = 0
    chunks_count: int = 0
    table_count: int = 0
    ids: List[str] = Field(default_factory=list)
    elapsed_ms: float = 0.0
    # 本次请求期间新增的向量缓存命中次数（秒传 / 重复分块时明显 > 0）
    embedding_cache_hits: int = 0
    # 内容相同但文件名不同时，返回已存在文件的文件名
    existing_file_name: Optional[str] = None
    # 仅“只解析不入库”时返回分块明细，避免大文件响应体过大
    chunks: Optional[List[Dict[str, Any]]] = None


class ParsePathRequest(BaseModel):
    """解析已有文件的请求体。"""

    file_path: str


# ======================================================================
# 知识库管理
# ======================================================================
class CollectionCreateRequest(BaseModel):
    """创建集合请求体。"""

    name: str = Field(..., min_length=1, description="集合名称")


class CollectionInfo(BaseModel):
    """集合信息。"""

    name: str
    count: int = 0
    distance: str = "cosine"
    files: int = 0
    embedding_model: str = ""
    # 集合中实际存储的向量维度（空集合为 None）
    embedding_dimension: Optional[int] = None
    # 当前向量模型输出维度
    model_dimension: Optional[int] = None
    # 两者是否一致；任一未知时为 None
    compatible: Optional[bool] = None


class FileInfo(BaseModel):
    """文件信息（按 file_name 聚合后的分块统计）。"""

    file_name: str
    namespace: str = ""
    chunks: int = 0
    tables: int = 0
    pages: List[int] = Field(default_factory=list)
    total_chars: int = 0
    size: Optional[int] = None
    sha256: Optional[str] = None
    uploaded_at: Optional[float] = None
    status: str = "success"
    error: str = ""


class DeleteFileResponse(BaseModel):
    """删除文件响应。"""

    deleted: bool
    file_name: str
    collection_name: str
    removed_chunks: int = 0


class DeleteCollectionResponse(BaseModel):
    """删除集合响应。"""

    deleted: bool
    collection_name: str


# ======================================================================
# 系统状态
# ======================================================================
class WarmupResponse(BaseModel):
    """向量模型预热响应。"""

    status: str = "ok"
    mode: str = ""
    model: str = ""
    dimension: int = 0
    elapsed_ms: float = 0.0
    cache: Dict[str, Any] = Field(default_factory=dict)


class CacheClearResponse(BaseModel):
    """清空向量缓存响应。"""

    cleared: bool = True
    cache: Dict[str, Any] = Field(default_factory=dict)
