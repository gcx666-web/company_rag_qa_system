# -*- coding: utf-8 -*-
"""chat_router.py —— RAG 问答接口。

流程：解析供应商 -> 检索（向量模型带缓存）-> 组装提示词 -> 调用 LLM -> 返回
答案与来源。相比原实现：

- 未知模型 / 未配置密钥返回 400，而不是未捕获的 ValueError（500）；
- 检索与 LLM 调用分别兜底，错误信息更明确；
- 请求里的 ``top_k`` / ``max_tokens`` 真正生效；
- 来源附带 file_name / page / chunk_type，前端可以展示可溯源的引用卡片。
"""

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from src.api.config import settings
from src.api.desp import get_llm, get_search_service
from src.api.models import ChatRequest, ChatResponse
from src.api.utils import embedding_dimension_hint, validate_collection_name
from src.llm.rag import generate_answer, generate_answer_stream
from src.retrieval.search_service import RetrievalConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["聊天对话"])


def _resolve_provider(raw_input: str) -> str | None:
    """把请求里的 model 字段解析成 LLM 供应商名。

    - 直接传供应商名（zhipu/qwen/siliconflow）时原样返回；
    - 传具体模型名时反查所属供应商；
    - 传空值时回退默认供应商（DEFAULT_LLM_MODEL 或第一个已配置的供应商）。
    """
    raw = str(raw_input or "").strip()

    if not raw:
        # 默认值可能是供应商名，也可能是模型名
        raw = str(settings.Default_LLM_MODEL or "").strip()
        if not raw:
            for name, config in settings.LLM_PROVIDER.items():
                if (config or {}).get("API_KEY"):
                    return name
            return None

    if raw in settings.LLM_PROVIDER:
        return raw

    for provider_name, config in settings.LLM_PROVIDER.items():
        if (config or {}).get("MODEL") == raw:
            return provider_name
    return None


def _jsonable_tokens(tokens):
    """把 openai 的 CompletionUsage 对象转成可序列化的字典。"""
    if tokens is None:
        return {}
    if isinstance(tokens, dict):
        return tokens
    for method in ("model_dump", "dict"):
        converter = getattr(tokens, method, None)
        if callable(converter):
            try:
                return converter()
            except Exception:
                pass
    return {}


def _build_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """构建来源卡片：保留文件名、页码、块类型，便于前端溯源。"""
    sources = []
    for result in results:
        metadata = result.get("metadata") or {}
        content = str(result.get("content") or "")
        sources.append(
            {
                "id": result.get("id"),
                "score": result.get("score"),
                "distance": result.get("distance"),
                "file_name": metadata.get("file_name") or "",
                "page": metadata.get("page"),
                "chunk_type": metadata.get("chunk_type") or "text",
                "content": content[:300] + "..." if len(content) > 300 else content,
            }
        )
    return sources


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    """序列化一个 SSE 事件帧。

    事件格式为 ``event: <名称>`` + 一行 ``data: <JSON>``，最后用空行分隔。
    JSON 里的换行会被转义，因此每个事件始终只占一行 ``data``，避免被 SSE
    按行拆坏；``ensure_ascii=False`` 保留中文可读性。
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


# 流式响应必须显式禁用各级缓存：StreamingResponse 本身不缓存，但浏览器和反向
# 代理（如 Nginx）可能把响应攒批后再下发，导致“流式”变成一次性输出。
_STREAM_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
    # 关闭 Nginx 的 proxy_buffering，保证逐块下发
    "X-Accel-Buffering": "no",
}


@router.post("/", response_model=ChatResponse, summary="RAG 问答")
async def chat(request: ChatRequest) -> ChatResponse:
    """基于指定知识库回答问题，并返回命中的原文片段作为来源。"""
    start_time = time.time()

    # 1) 解析供应商并获取缓存的 LLM 客户端
    provider = _resolve_provider(request.model)
    if provider is None:
        configured = [
            name
            for name, config in settings.LLM_PROVIDER.items()
            if (config or {}).get("API_KEY")
        ]
        raise HTTPException(
            status_code=400,
            detail=(
                f"未知模型：{request.model!r}。可用的供应商："
                f"{configured or '无（请先在 .env 配置 LLM 密钥）'}"
            ),
        )
    try:
        llm = get_llm(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2) 检索（向量模型命中缓存时不会重复调用向量接口）
    collection_name = validate_collection_name(request.collection_name or "default")
    search_service = get_search_service(collection_name)
    search_config = RetrievalConfig(top_k=request.top_k)
    try:
        # 检索包含 Chroma 查询与向量计算（可能走网络），放到线程池避免阻塞事件循环
        search_response = await run_in_threadpool(
            search_service.search, request.question, search_config
        )
    except Exception as exc:
        logger.exception(
            "检索失败：collection=%s, question=%s", collection_name, request.question
        )
        # 维度不匹配属于配置问题而非服务故障，返回 409 让前端直接展示处理建议
        hint = embedding_dimension_hint(exc, collection_name)
        if hint:
            raise HTTPException(status_code=409, detail=hint) from exc
        raise HTTPException(
            status_code=500, detail=f"知识库检索失败: {exc}"
        ) from exc

    # 3) 生成答案
    try:
        answer = generate_answer(
            llm=llm,
            question=request.question,
            result=search_response.results,
            temperature=settings.LLM_TEMPERATURE,
            max_token=request.max_tokens,
        )
    except Exception as exc:
        logger.exception(
            "LLM 调用失败：provider=%s, model=%s", provider, getattr(llm, "model", "")
        )
        raise HTTPException(
            status_code=502, detail=f"大模型调用失败: {exc}"
        ) from exc

    # 4) 构建来源卡片：保留文件名、页码、块类型，便于前端溯源
    sources = _build_sources(search_response.results)

    total_ms = (time.time() - start_time) * 1000
    return ChatResponse(
        question=answer.get("question", request.question),
        answer=answer.get("answer", ""),
        context=answer.get("context", ""),
        tokens_used=_jsonable_tokens(answer.get("tokens_used")),
        sources=sources,
        total_time_ms=round(total_ms, 2),
        retrieval_time_ms=round(float(search_response.time), 2),
        model=str(getattr(llm, "model", "")),
        collection_name=collection_name,
    )


@router.post("/stream", summary="RAG 问答（SSE 流式）")
async def chat_stream(request: ChatRequest, raw_request: Request) -> StreamingResponse:
    """基于指定知识库流式回答问题（Server-Sent Events）。

    事件类型（``event`` 字段）：

    - ``meta``：问题、上下文、来源、模型、检索耗时（流开始时推送一次）；
    - ``delta``：增量答案文本，前端拼接后即可实现逐字输出；
    - ``done``：完整答案、结束原因、token 用量、总耗时；
    - ``error``：流式过程中的错误（此时 200 响应头已发出，无法再改 HTTP 状态码）。

    与 ``POST /chat/`` 的区别：校验与检索仍同步完成，所以未知模型、集合非法、
    向量维度不一致等问题依旧返回 400 / 409 / 500；之后的答案生成改为异步流式，
    并通过响应头显式禁用各级缓存，确保逐块下发。
    """
    start_time = time.time()

    # 1) 解析供应商并获取缓存的 LLM 客户端（与 /chat/ 保持一致）
    provider = _resolve_provider(request.model)
    if provider is None:
        configured = [
            name
            for name, config in settings.LLM_PROVIDER.items()
            if (config or {}).get("API_KEY")
        ]
        raise HTTPException(
            status_code=400,
            detail=(
                f"未知模型：{request.model!r}。可用的供应商："
                f"{configured or '无（请先在 .env 配置 LLM 密钥）'}"
            ),
        )
    try:
        llm = get_llm(provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # 2) 检索放到线程池，避免 Chroma / 向量模型阻塞事件循环
    collection_name = validate_collection_name(request.collection_name or "default")
    search_service = get_search_service(collection_name)
    search_config = RetrievalConfig(top_k=request.top_k)
    try:
        search_response = await run_in_threadpool(
            search_service.search, request.question, search_config
        )
    except Exception as exc:
        logger.exception(
            "流式问答检索失败：collection=%s, question=%s",
            collection_name,
            request.question,
        )
        hint = embedding_dimension_hint(exc, collection_name)
        if hint:
            raise HTTPException(status_code=409, detail=hint) from exc
        raise HTTPException(
            status_code=500, detail=f"知识库检索失败: {exc}"
        ) from exc

    sources = _build_sources(search_response.results)
    retrieval_time_ms = round(float(search_response.time), 2)
    model_name = str(getattr(llm, "model", ""))

    async def event_stream() -> AsyncIterator[str]:
        """把 RAG 生成器的事件转换成 SSE 帧。"""
        answer_parts: List[str] = []
        final_answer = ""
        finish_reason = None
        tokens_used: Dict[str, Any] = {}
        disconnected = False

        generator = generate_answer_stream(
            llm=llm,
            question=request.question,
            result=search_response.results,
            temperature=settings.LLM_TEMPERATURE,
            max_token=request.max_tokens,
        )
        try:
            # 先取 start 事件，把格式化后的上下文随 meta 一起推给前端
            start_event = await generator.__anext__()
            yield _sse_event(
                "meta",
                {
                    "question": request.question,
                    "context": start_event.get("context", ""),
                    "sources": sources,
                    "model": model_name or start_event.get("model", ""),
                    "collection_name": collection_name,
                    "retrieval_time_ms": retrieval_time_ms,
                },
            )

            async for event in generator:
                # 每块前检查客户端是否已断开；断开后立即停止，避免继续消耗模型额度
                if await raw_request.is_disconnected():
                    disconnected = True
                    logger.info(
                        "客户端已断开，停止流式生成：collection=%s", collection_name
                    )
                    break

                event_type = event.get("type")
                if event_type == "delta":
                    chunk = str(event.get("content") or "")
                    answer_parts.append(chunk)
                    yield _sse_event("delta", {"content": chunk})
                elif event_type == "done":
                    final_answer = str(event.get("answer") or "")
                    finish_reason = event.get("finish_reason")
                    tokens_used = _jsonable_tokens(event.get("tokens_used"))

            if disconnected:
                return

            total_ms = (time.time() - start_time) * 1000
            yield _sse_event(
                "done",
                {
                    "answer": final_answer or "".join(answer_parts).strip(),
                    "finish_reason": finish_reason,
                    "tokens_used": tokens_used,
                    "total_time_ms": round(total_ms, 2),
                    "retrieval_time_ms": retrieval_time_ms,
                    "model": model_name,
                    "collection_name": collection_name,
                },
            )
        except StopAsyncIteration:
            logger.error("流式生成未返回起始事件：collection=%s", collection_name)
            yield _sse_event("error", {"message": "流式生成未返回起始事件"})
        except asyncio.CancelledError:
            # 客户端断开 / 服务关闭时由框架取消任务，不属于业务错误
            raise
        except Exception as exc:
            logger.exception(
                "流式生成失败：provider=%s, model=%s, collection=%s",
                provider,
                model_name,
                collection_name,
            )
            yield _sse_event("error", {"message": f"大模型调用失败: {exc}"})
        finally:
            # 无论正常结束还是中途断开，都关闭生成器，触发上游 AsyncStream.close()
            try:
                await generator.aclose()
            except BaseException:  # 取消/断连时关闭失败可忽略，连接会随请求回收
                pass

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=_STREAM_HEADERS,
    )
