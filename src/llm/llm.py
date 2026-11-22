# -*- coding: utf-8 -*-
"""LLM 客户端封装。

说明：``openai.OpenAI`` 内部维护连接池，创建成本比单次请求高，因此实例由
``src.api.desp.get_llm`` 按供应商缓存复用，这里只负责构造与调用。新增
``timeout`` 参数，避免上游接口卡住时 FastAPI 请求无限等待。
"""

import inspect
from typing import Any, AsyncIterator, Dict, List

import openai


class LLM:
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-ai/DeepSeek-V4.1-Flash",
        base_url: str = "https://api.deepseek.ai/v1/",
        timeout: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        # 客户端只创建一次（由上层缓存本实例），后续请求复用连接池。
        # 只有显式传入 timeout 时才增加该参数，保持旧调用的参数完全不变。
        self.client = openai.OpenAI(**self._client_kwargs())
        # 异步客户端延迟创建：只有流式接口会用到，非流式请求无需多建连接池
        self._async_client: openai.AsyncOpenAI | None = None

    def _client_kwargs(self) -> Dict[str, Any]:
        """同步 / 异步客户端共用的构造参数（timeout 仅在显式传入时下发）。"""
        client_kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        return client_kwargs

    @property
    def async_client(self) -> openai.AsyncOpenAI:
        """异步客户端：首次访问时创建，之后复用同一连接池。"""
        if self._async_client is None:
            self._async_client = openai.AsyncOpenAI(**self._client_kwargs())
        return self._async_client

    def generate(
        self,
        message: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """生成内容。

        :param message:     聊天消息列表（OpenAI 格式）。
        :param temperature: 采样温度。
        :param max_tokens:  最大生成 token 数。
        :return: 包含 content / finish_reason / token_usage 的字典。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            'content': response.choices[0].message.content,
            'finish_reason': response.choices[0].finish_reason,
            # CompletionUsage 是对象而非 dict，响应模型里用 Any 接收
            'token_usage': response.usage,
        }

    async def _create_stream(
        self,
        message: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        include_usage: bool,
    ):
        """发起一次流式请求，返回 OpenAI 的 AsyncStream 对象。"""
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": message,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if include_usage:
            # 让兼容 OpenAI 的供应商在最后一块返回 token 用量
            kwargs["stream_options"] = {"include_usage": True}
        return await self.async_client.chat.completions.create(**kwargs)

    async def generate_stream(
        self,
        message: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> AsyncIterator[Dict[str, Any]]:
        """异步流式生成内容（逐块产出增量文本）。

        产出的事件：

        - ``{"type": "delta", "content": "..."}``：增量文本，空增量会被过滤；
        - ``{"type": "done", "content": 完整文本, "finish_reason": ..., "token_usage": ...}``：
          流结束时给出的汇总信息。

        兼容性：``stream_options`` 只有部分供应商支持，这里先带上以获取 token
        用量，遇到 400 时自动去掉该参数重试一次，保证流式能力不被供应商差异打断。
        """
        try:
            stream = await self._create_stream(
                message, temperature, max_tokens, include_usage=True
            )
        except openai.BadRequestError:
            stream = await self._create_stream(
                message, temperature, max_tokens, include_usage=False
            )

        parts: List[str] = []
        finish_reason = None
        token_usage = None
        try:
            async for chunk in stream:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    token_usage = usage
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    # 只携带 usage 的收尾块没有 choices
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if content:
                    parts.append(content)
                    yield {"type": "delta", "content": content}
                choice_finish = getattr(choice, "finish_reason", None)
                if choice_finish:
                    finish_reason = choice_finish
        finally:
            # 客户端中途断开 / 迭代出错时也要关闭上游连接，避免连接泄漏
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    result = close()
                    if inspect.isawaitable(result):
                        await result
                except Exception:  # 关闭失败不影响已经产出的内容
                    pass

        yield {
            "type": "done",
            "content": "".join(parts),
            "finish_reason": finish_reason,
            "token_usage": token_usage,
        }
