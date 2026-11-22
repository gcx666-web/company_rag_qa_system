from fastapi import APIRouter, Depends, HTTPException, status
from src.api.models import ChatRequest, ChatResponse
from src.api.config import settings
from src.api.desp import get_llm
from src.api.desp import get_search_service
from src.retrieval.search_service import RetrievalConfig
from src.llm.rag import build_prompt, format_context, generate_answer
import time
router = APIRouter(prefix="/chat", tags=["聊天对话"])
@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    '''rag聊天对话'''
    #进入对话获取时间
    start_time = time.time()
    raw_input = request.model or settings.Default_LLM_MODEL

    # 判断：传入的是不是provider
    if raw_input in settings.LLM_PROVIDER:
        provider = raw_input
    else:
        # 是具体模型名，循环查找属于哪个provider
        provider = None
        for p_name, cfg in settings.LLM_PROVIDER.items():
            if cfg.get("MODEL") == raw_input:
                provider = p_name
                break
        if provider is None:
            raise ValueError(f"model {raw_input} 不存在，不是服务商也不是已配置模型")

    llm = get_llm(provider)
    #获取检索服务实例
    search_service = get_search_service(request.collection_name)
    #创建检索配置对象
    search_config=RetrievalConfig(
        top_k=settings.TOP_K,
    )
    #检索
    search_response=search_service.search(request.question,search_config)
    #生成答案
    answer=generate_answer(
        llm=llm,
        question=request.question,
        result=search_response.results,
        temperature=settings.LLM_TEMPERATURE,
        max_token=settings.LLM_MAX_TOKENS,
    )
    #构建来源
    sources=[
        {
            "id": result["id"],
            
            "score": result["score"],
            "content": result["content"][:200]+'...' if len(result["content"])>200 else result["content"],
            
        }
        for result in search_response.results
    ]
    #结束获取时间
    end_time = time.time()
    total=(end_time - start_time)*1000
    return ChatResponse(
        question=answer["question"],
        answer=answer["answer"],
        context=answer["context"],
        tokens_used=answer["tokens_used"],
        sources=sources,
        total_time_ms=total,
    )
    