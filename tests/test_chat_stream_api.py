# -*- coding: utf-8 -*-
"""流式问答接口测试：SSE 事件、缓存禁用响应头与异常处理（全部 mock，不联网）。

运行方式（项目根目录下）::

    .venv/Scripts/python.exe -m unittest tests.test_chat_stream_api -v
"""

import json
import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routers import chat_router
from src.retrieval.search_service import SearchResponse


def parse_sse(text):
    """把响应文本拆成 [(事件名, data 字典)]，忽略空帧。"""
    events = []
    for frame in text.split("\n\n"):
        if not frame.strip():
            continue
        name = "message"
        data_lines = []
        for line in frame.splitlines():
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())
        if not data_lines:
            continue
        events.append((name, json.loads("\n".join(data_lines))))
    return events


class FakeStreamLLM:
    """假 LLM：同时实现同步 generate 与异步 generate_stream，便于两种接口测试。"""

    model = "fake-stream-model"

    def __init__(self, chunks=("公司年假", "为15天。"), error=None):
        self.chunks = list(chunks)
        self.error = error
        self.stream_calls = []
        self.generate_calls = []

    def generate(self, message, temperature=0.7, max_tokens=2000):
        self.generate_calls.append(
            {"message": message, "temperature": temperature, "max_tokens": max_tokens}
        )
        return {
            "content": "".join(self.chunks),
            "finish_reason": "stop",
            "token_usage": {},
        }

    async def generate_stream(self, message, temperature=0.7, max_tokens=2000):
        self.stream_calls.append(
            {"message": message, "temperature": temperature, "max_tokens": max_tokens}
        )
        for chunk in self.chunks:
            yield {"type": "delta", "content": chunk}
        if self.error is not None:
            raise self.error
        yield {
            "type": "done",
            "content": "".join(self.chunks),
            "finish_reason": "stop",
            "token_usage": {},
        }


class FakeSearchService:
    """假检索服务：返回预设结果，或抛出指定异常。"""

    def __init__(self, results=None, error=None):
        self.results = [] if results is None else results
        self.error = error
        self.calls = []

    def search(self, query, config=None):
        self.calls.append({"query": query, "config": config})
        if self.error is not None:
            raise self.error
        return SearchResponse(
            query=query,
            total=len(self.results),
            results=self.results,
            time=12.34,
        )


class ChatStreamTestBase(unittest.TestCase):
    """统一准备：假 LLM、假检索服务、临时供应商配置。"""

    payload = {
        "question": "公司年假有多少天？",
        "model": "demo",
        "collection_name": "default",
        "top_k": 3,
    }

    def setUp(self):
        self.client = TestClient(app)
        self.results = [
            {
                "id": "c1",
                "content": "公司员工年假为15天",
                "score": 0.92,
                "distance": 0.08,
                "metadata": {
                    "file_name": "员工手册.pdf",
                    "page": 3,
                    "chunk_type": "text",
                },
            },
            {
                "id": "c2",
                "content": "请假需提前3天申请",
                "score": 0.8,
                "distance": 0.2,
                "metadata": {},
            },
        ]
        self.search_service = FakeSearchService(results=self.results)
        self.llm = FakeStreamLLM()
        patches = [
            mock.patch.object(
                chat_router.settings,
                "LLM_PROVIDER",
                {
                    "demo": {
                        "API_KEY": "sk-test",
                        "BASE_URL": "https://example.com/v1",
                        "MODEL": "demo-model",
                    }
                },
            ),
            mock.patch.object(chat_router, "get_llm", lambda provider: self.llm),
            mock.patch.object(
                chat_router, "get_search_service", lambda name: self.search_service
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def stream(self, payload=None):
        """请求流式接口，返回 (状态码, 响应头, 完整响应文本)。"""
        body = self.payload if payload is None else payload
        with self.client.stream("POST", "/api/chat/stream", json=body) as response:
            return response.status_code, dict(response.headers), "".join(
                response.iter_text()
            )


class ChatStreamAPITest(ChatStreamTestBase):
    """流式接口：事件协议、缓存禁用头、错误分支。"""

    def test_流式事件与禁用缓存响应头(self):
        status, headers, text = self.stream()

        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers.get("content-type", ""))

        # 关键要求：流式响应必须禁用各级缓存，避免被攒批后一次性下发
        cache_control = headers.get("cache-control", "")
        self.assertIn("no-store", cache_control)
        self.assertIn("no-cache", cache_control)
        self.assertIn("must-revalidate", cache_control)
        self.assertIn("max-age=0", cache_control)
        self.assertEqual(headers.get("pragma"), "no-cache")
        self.assertEqual(headers.get("expires"), "0")
        self.assertEqual(headers.get("x-accel-buffering"), "no")

        events = parse_sse(text)
        names = [name for name, _ in events]
        self.assertEqual(names[0], "meta")
        self.assertEqual(names[-1], "done")
        self.assertEqual(names.count("delta"), 2)
        self.assertNotIn("error", names)

        meta = events[0][1]
        self.assertEqual(meta["question"], "公司年假有多少天？")
        self.assertIn("公司员工年假为15天", meta["context"])
        self.assertEqual(meta["model"], "fake-stream-model")
        self.assertEqual(meta["collection_name"], "default")
        self.assertAlmostEqual(meta["retrieval_time_ms"], 12.34, places=2)
        self.assertEqual(len(meta["sources"]), 2)
        self.assertEqual(meta["sources"][0]["file_name"], "员工手册.pdf")
        self.assertEqual(meta["sources"][0]["page"], 3)

        deltas = [data["content"] for name, data in events if name == "delta"]
        self.assertEqual(deltas, ["公司年假", "为15天。"])

        done = events[-1][1]
        self.assertEqual(done["answer"], "公司年假为15天。")
        self.assertEqual(done["finish_reason"], "stop")
        self.assertEqual(done["model"], "fake-stream-model")
        self.assertEqual(done["collection_name"], "default")
        self.assertIn("total_time_ms", done)
        self.assertIn("retrieval_time_ms", done)

        # 检索参数透传，提示词包含检索到的上下文
        self.assertEqual(self.search_service.calls[0]["query"], "公司年假有多少天？")
        self.assertEqual(self.search_service.calls[0]["config"].top_k, 3)
        prompt = self.llm.stream_calls[0]["message"][0]["content"]
        self.assertIn("公司年假有多少天？", prompt)
        self.assertIn("公司员工年假为15天", prompt)

    def test_空检索结果不调用模型(self):
        self.search_service.results = []
        status, _, text = self.stream()

        self.assertEqual(status, 200)
        events = parse_sse(text)
        names = [name for name, _ in events]
        self.assertEqual(names, ["meta", "delta", "done"])
        self.assertEqual(events[1][1]["content"], "没有找到答案")
        self.assertEqual(events[-1][1]["answer"], "没有找到答案")
        self.assertEqual(events[-1][1]["finish_reason"], "empty_result")
        self.assertEqual(self.llm.stream_calls, [])

    def test_流中异常推送error事件(self):
        self.llm.error = RuntimeError("上游断流")
        status, _, text = self.stream()

        # 响应头已经发出，业务错误只能通过 SSE 事件传递
        self.assertEqual(status, 200)
        events = parse_sse(text)
        names = [name for name, _ in events]
        self.assertEqual(names[0], "meta")
        # 已经产出的增量仍然保留
        self.assertEqual(names.count("delta"), 2)
        self.assertEqual(names[-1], "error")
        self.assertIn("上游断流", events[-1][1]["message"])

    def test_未知模型返回400(self):
        payload = dict(self.payload, model="not-exist")
        response = self.client.post("/api/chat/stream", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("未知模型", response.json()["detail"])

    def test_向量维度不匹配返回409(self):
        self.search_service.error = Exception(
            "Collection expecting embedding with dimension of 512, got 1024"
        )
        response = self.client.post("/api/chat/stream", json=self.payload)

        self.assertEqual(response.status_code, 409)
        self.assertIn("向量维度", response.json()["detail"])

    def test_非流式接口仍保持兼容(self):
        response = self.client.post("/api/chat/", json=self.payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["answer"], "公司年假为15天。")
        self.assertEqual(len(body["sources"]), 2)
        self.assertEqual(body["model"], "fake-stream-model")
        self.assertEqual(self.search_service.calls[0]["query"], "公司年假有多少天？")

