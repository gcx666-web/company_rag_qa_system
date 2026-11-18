"""API 请求/响应模型。"""

from gc import collect
from typing import Any

from pydantic import BaseModel, Field

from src.api.config import settings


class ChatRequest(BaseModel):
    """问答请求模型"""

    question: str = Field(..., min_length=1, description="问题")
    max_tokens: int = Field(
        default=settings.LLM_MAX_TOKENS,
        gt=0,
        description="最大生成长度",
    )
    model: str = Field(
        default=settings.Default_LLM_MODEL or "",
        description="模型名称",
    )
    collection_name: str = Field(
        default="default",
        description="向量库名称",
    )


class ChatResponse(BaseModel):
    """问答响应模型，字段与 src.llm.rag.generate_answer 的返回结构对齐。

    说明：
    tokens_used 直接来自 openai 的 CompletionUsage 对象（不是 dict），
    因此声明为 Any，保证正常分支与空结果分支都能直接构造并序列化。
    """

    answer: str = Field(..., description="答案")
    question: str = Field(..., description="问题")
    context: str = Field(default="", description="拼装后的上下文")
    tokens_used: Any = Field(default_factory=dict, description="本次调用的大模型 token 用量")
    model: str = Field(default="", description="回答所用模型")

