# -*- coding: utf-8 -*-
"""SearchService 检索服务的测试（Stub 单元测试 + 真实 Chroma 集成测试）。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    .venv/Scripts/python.exe -m unittest tests.test_search_service -v

也可以直接运行本文件::

    .venv/Scripts/python.exe tests/test_search_service.py

说明：
1. Stub 部分用一个只实现 search 的假 DocumentStore，记录入参并返回预设结果，
   验证 SearchService 的编排逻辑（默认配置、top_k / namespace 透传、min_score
   过滤、二次截断、异常透传），不落盘、不联网、毫秒级；
2. 集成部分使用临时目录创建真实 PersistentClient + 假向量模型（字符 ord 落桶，
   不下载模型），验证 DocumentStore → ChromaVectorStore 的真实检索链路；
3. Windows 下 Chroma 会占用 sqlite/hnsw 文件，tearDown 先清 Chroma 的系统缓存
   再删临时目录，删除失败不视为测试失败（rmtree(ignore_errors=True)）；
4. 用标准库 unittest 编写，无需额外安装 pytest。
"""

import gc
import os
import shutil
import sys
import tempfile
import unittest

# 把项目根目录加入 sys.path，保证用任意方式、从任意目录运行都能导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.document_store import DocumentStore
from src.embeddings.base import Base
from src.embeddings.chorm_store import ChromaVectorStore
from src.parsers.base import ParseResult
from src.retrieval.search_service import (
    RetrievalConfig,
    SearchResponse,
    SearchService,
)


class FakeEmbedding(Base):
    """确定性假向量模型：按字符 ord 落到固定维度桶，不下载模型、不联网。"""

    def __init__(self, dimension: int = 16, model_name: str = "fake-embedding"):
        self.dimension = dimension
        self.model_name = model_name
        # 记录每次批量嵌入的入参，便于断言“整批只调用一次”
        self.batch_calls = []

    def embed_text(self, text: str) -> list:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list) -> list:
        self.batch_calls.append(list(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimension
            for ch in str(text):
                vector[ord(ch) % self.dimension] += 1.0
            vectors.append(vector)
        return vectors

    def get_dimension(self) -> int:
        return self.dimension

    def get_model_name(self) -> str:
        return self.model_name


class StubDocumentStore:
    """只实现 search 的假 DocumentStore：记录入参、按预设结果返回、可注入异常。

    真实 DocumentStore.search 返回的每条形如
    ``{"id", "content", "score", "distance", "metadata"}``，这里保持同一契约，
    这样 SearchService 按 ``result["score"]`` 过滤时行为一致。
    """

    def __init__(self, results=None, error=None):
        self.results = [] if results is None else list(results)
        self.error = error
        # 每次调用的入参记录：{"query": ..., "top_k": ..., "namespace": ...}
        self.calls = []

    def search(self, query, top_k=5, namespace=None):
        self.calls.append(
            {"query": query, "top_k": top_k, "namespace": namespace}
        )
        if self.error is not None:
            raise self.error
        # 返回新列表但复用同一个结果字典对象，便于断言“原样透传”
        return list(self.results)

    @property
    def last_call(self):
        return self.calls[-1]


def make_result(
    document_id="doc-1",
    content="测试内容",
    score=0.9,
    distance=None,
    metadata=None,
) -> dict:
    """构造与 DocumentStore.search 返回值同构的结果字典。"""
    return {
        "id": document_id,
        "content": content,
        "score": score,
        "distance": distance,
        "metadata": {} if metadata is None else metadata,
    }


def make_parse_result(chunks, file_name="hr.md") -> ParseResult:
    """构造测试用 ParseResult；DocumentStore 只用到 file_name 与 chunks。"""
    return ParseResult(
        file_name=file_name,
        total_pages=1,
        total_chunks=len(chunks),
        table_count=0,
        chunks=chunks,
        raw_text="",
        tables=[],
        metadata={},
    )


# ======================================================================
# 一、RetrievalConfig / SearchResponse：数据结构契约
# ======================================================================
class TestRetrievalConfigAndResponse(unittest.TestCase):
    """两个 dataclass 的默认值与字段保存行为。"""

    def test_RetrievalConfig默认值(self):
        config = RetrievalConfig()

        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.min_score, 0.0)
        self.assertIsNone(config.namespace)
        self.assertIsNone(config.metadata_filter)

    def test_RetrievalConfig可自定义(self):
        config = RetrievalConfig(
            top_k=3,
            min_score=0.6,
            namespace="hr",
            metadata_filter={"source": "wiki"},
        )

        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.min_score, 0.6)
        self.assertEqual(config.namespace, "hr")
        self.assertEqual(config.metadata_filter, {"source": "wiki"})

    def test_SearchResponse字段与默认time(self):
        results = [make_result("d1")]
        response = SearchResponse(query="年假申请", total=1, results=results)

        self.assertEqual(response.query, "年假申请")
        self.assertEqual(response.total, 1)
        self.assertEqual(response.results, results)
        # time 未显式传入时默认 0.0（毫秒）
        self.assertEqual(response.time, 0.0)


# ======================================================================
# 二、SearchService 编排逻辑：用假 DocumentStore 验证
# ======================================================================
class TestSearchServiceWithStubStore(unittest.TestCase):
    """只验证 SearchService 自身的编排，不创建真实数据库与向量模型。"""

    def _make_service(self, results=None, error=None):
        """构造 SearchService + 记录入参的假 DocumentStore。"""
        store = StubDocumentStore(results=results, error=error)
        return SearchService(document_store=store), store

    def test_构造时保存document_store(self):
        service, store = self._make_service()

        self.assertIs(service.document_store, store)

    def test_search_不传config时使用默认配置(self):
        results = [make_result("hr_hr.md_0", "年假申请流程", 0.9)]
        service, store = self._make_service(results)

        response = service.search("年假申请")

        # 默认 RetrievalConfig：top_k=5、namespace=None
        self.assertEqual(
            store.calls,
            [{"query": "年假申请", "top_k": 5, "namespace": None}],
        )
        self.assertEqual(response.query, "年假申请")
        self.assertEqual(response.total, 1)
        self.assertEqual(response.results, results)

    def test_search_透传top_k与namespace(self):
        service, store = self._make_service([make_result("life_life.md_0")])

        service.search("食堂菜单", RetrievalConfig(top_k=2, namespace="life"))

        self.assertEqual(store.last_call["query"], "食堂菜单")
        self.assertEqual(store.last_call["top_k"], 2)
        self.assertEqual(store.last_call["namespace"], "life")

    def test_search_按min_score过滤且包含边界(self):
        # 分数依次 0.9 / 0.5 / 0.1，阈值 0.5：等于阈值的记录必须保留（>= 而非 >）
        results = [
            make_result("d1", score=0.9),
            make_result("d2", score=0.5),
            make_result("d3", score=0.1),
        ]
        service, _ = self._make_service(results)

        response = service.search("问题", RetrievalConfig(min_score=0.5))

        self.assertEqual([r["id"] for r in response.results], ["d1", "d2"])
        self.assertEqual(response.total, 2)

    def test_search_结果被截断到top_k(self):
        # 假向量库故意超额返回 4 条，验证 SearchService 自己的二次截断生效
        results = [make_result(f"d{i}", score=0.9 - i * 0.1) for i in range(4)]
        service, _ = self._make_service(results)

        response = service.search("问题", RetrievalConfig(top_k=2))

        self.assertEqual([r["id"] for r in response.results], ["d0", "d1"])
        self.assertEqual(response.total, 2)

    def test_search_结果字典原样透传(self):
        results = [make_result("d1", "原文内容", 0.8, 0.2, {"namespace": "hr"})]
        service, _ = self._make_service(results)

        response = service.search("问题")

        # 不重建字典：调用方拿到的就是底层返回的对象
        self.assertIs(response.results[0], results[0])
        self.assertEqual(response.results[0]["distance"], 0.2)
        self.assertEqual(response.results[0]["metadata"], {"namespace": "hr"})

    def test_search_空结果时total为0(self):
        service, store = self._make_service([])

        response = service.search("问题")

        self.assertEqual(response.results, [])
        self.assertEqual(response.total, 0)
        # 仍然调用了底层，只是没拿到记录
        self.assertEqual(len(store.calls), 1)

    def test_search_time为非负毫秒(self):
        service, _ = self._make_service([make_result("d1")])

        response = service.search("问题")

        self.assertIsInstance(response.time, float)
        self.assertGreaterEqual(response.time, 0.0)

    def test_search_query原样透传(self):
        service, store = self._make_service([])

        # SearchService 不做入参校验，空白 query 也原样交给 DocumentStore
        response = service.search("   ")

        self.assertEqual(store.last_call["query"], "   ")
        self.assertEqual(response.query, "   ")

    def test_search_metadata_filter当前不传给底层(self):
        service, store = self._make_service([])

        service.search("问题", RetrievalConfig(metadata_filter={"source": "wiki"}))

        # 现状：DocumentStore.search 不支持元数据过滤，metadata_filter 暂未实现，
        # 底层只会收到 query / top_k / namespace；后续支持时请同步修改本用例。
        self.assertEqual(set(store.last_call.keys()), {"query", "top_k", "namespace"})

    def test_search_底层异常向上抛出(self):
        service, _ = self._make_service(error=RuntimeError("向量库不可用"))

        # SearchService 不吞异常，交由上层（API / 应用层）决定如何兜底
        with self.assertRaises(RuntimeError):
            service.search("问题")


# ======================================================================
# 三、集成测试：真实 Chroma（临时目录）+ 假向量模型
# ======================================================================
class TestSearchServiceIntegration(unittest.TestCase):
    """验证 DocumentStore → ChromaVectorStore → SearchService 的端到端检索。"""

    def setUp(self):
        from chromadb.api.shared_system_client import SharedSystemClient

        self.addCleanup(self._cleanup_tempdir)
        self.tmpdir = tempfile.mkdtemp(prefix="search_service_test_")
        self.vector_store = ChromaVectorStore(
            collection_name="test_search_service",
            persist_directory=self.tmpdir,
        )
        self.embedding = FakeEmbedding()
        self.doc_store = DocumentStore(
            embedding_model=self.embedding, vector_store=self.vector_store
        )
        self.service = SearchService(document_store=self.doc_store)
        # cleanup 是 LIFO：先清 Chroma 的系统缓存，再删临时目录
        self.addCleanup(SharedSystemClient.clear_system_cache)

    def _cleanup_tempdir(self):
        # Windows 下 Chroma 会占用文件，先 GC 再删；删不掉也不让测试失败
        self.service = None
        self.doc_store = None
        self.vector_store = None
        gc.collect()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _add_hr_and_life_docs(self):
        """写入两条不同 namespace 的文档：hr / life。"""
        self.doc_store.add_parse_result(
            make_parse_result(
                [{"chunk_id": 0, "type": "text", "content": "年假申请流程"}],
                file_name="hr.md",
            ),
            namespace="hr",
        )
        self.doc_store.add_parse_result(
            make_parse_result(
                [{"chunk_id": 0, "type": "text", "content": "食堂菜单介绍"}],
                file_name="life.md",
            ),
            namespace="life",
        )

    def test_search_默认配置返回降序结果(self):
        self._add_hr_and_life_docs()

        response = self.service.search("年假申请")

        self.assertEqual(response.query, "年假申请")
        self.assertEqual(response.total, 2)
        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.results[0]["id"], "hr_hr.md_0")
        self.assertEqual(response.results[0]["content"], "年假申请流程")
        self.assertGreater(response.results[0]["score"], response.results[1]["score"])
        self.assertIsNotNone(response.results[0]["distance"])
        self.assertEqual(response.results[0]["metadata"]["namespace"], "hr")
        self.assertEqual(
            set(response.results[0].keys()),
            {"id", "content", "score", "distance", "metadata"},
        )
        self.assertGreaterEqual(response.time, 0.0)

    def test_search_按namespace过滤(self):
        self._add_hr_and_life_docs()

        response = self.service.search("年假申请", RetrievalConfig(namespace="life"))

        self.assertEqual([r["id"] for r in response.results], ["life_life.md_0"])
        self.assertEqual(response.total, 1)
        self.assertEqual(response.results[0]["metadata"]["namespace"], "life")

    def test_search_按top_k截断(self):
        self.doc_store.add_documents(
            ["年假申请流程", "年假申请流程说明", "食堂菜单介绍"],
            [{}, {}, {}],
            namespace="hr",
        )

        response = self.service.search("年假申请", RetrievalConfig(top_k=2))

        self.assertEqual(len(response.results), 2)
        self.assertEqual(response.total, 2)

    def test_search_min_score过滤(self):
        self._add_hr_and_life_docs()
        # 先查一次拿到真实分数，避免在测试里硬编码向量模型算出的具体分值
        all_response = self.service.search("年假申请")
        lowest = min(r["score"] for r in all_response.results)
        threshold = lowest + 1e-6

        response = self.service.search(
            "年假申请", RetrievalConfig(min_score=threshold)
        )

        self.assertEqual(response.total, 1)
        self.assertEqual(response.results[0]["id"], "hr_hr.md_0")
        self.assertTrue(all(r["score"] >= threshold for r in response.results))

    def test_search_空集合返回空结果(self):
        response = self.service.search("任意问题")

        self.assertEqual(response.results, [])
        self.assertEqual(response.total, 0)
        self.assertGreaterEqual(response.time, 0.0)
        # 带 namespace 的过滤路径同样不越界
        self.assertEqual(
            self.service.search("任意问题", RetrievalConfig(namespace="hr")).total, 0
        )

    def test_search_空白query返回空结果(self):
        self._add_hr_and_life_docs()

        with self.assertLogs("src.database.document_store", level="WARNING"):
            response = self.service.search("   ")

        self.assertEqual(response.results, [])
        self.assertEqual(response.total, 0)

    def test_search_清空后仍返回空结果(self):
        self._add_hr_and_life_docs()
        self.assertTrue(self.doc_store.clear())

        response = self.service.search("年假申请")

        self.assertEqual(response.total, 0)
        self.assertEqual(response.results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

