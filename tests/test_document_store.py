# -*- coding: utf-8 -*-
"""DocumentStore 文档存储服务的测试（假向量模型 + 真实 Chroma 临时目录）。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    .venv/Scripts/python.exe -m unittest tests.test_document_store -v

也可以直接运行本文件::

    .venv/Scripts/python.exe tests/test_document_store.py

说明：
1. FakeEmbedding 用“字符 ord 落桶”生成确定性向量，不下载模型、不联网；
   相同/相近文本得到相近向量，足以验证写入与检索链路；
2. 使用临时目录创建真实 PersistentClient，验证 add_parse_result / add_documents /
   search / get_stats / delete / clear 的实际行为；
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

from chromadb.errors import DuplicateIDError

from src.database.document_store import DocumentStore
from src.embeddings.base import Base
from src.embeddings.chorm_store import ChromaVectorStore
from src.parsers.base import ParseResult


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


class TestDocumentStore(unittest.TestCase):
    """使用真实 Chroma（临时目录）+ 假向量模型，验证 DocumentStore 全链路。"""

    def setUp(self):
        from chromadb.api.shared_system_client import SharedSystemClient

        self.addCleanup(self._cleanup_tempdir)
        self.tmpdir = tempfile.mkdtemp(prefix="document_store_test_")
        self.store = ChromaVectorStore(
            collection_name="test_document_store", persist_directory=self.tmpdir
        )
        self.embedding = FakeEmbedding()
        self.doc_store = DocumentStore(
            embedding_model=self.embedding, vector_store=self.store
        )
        self.addCleanup(SharedSystemClient.clear_system_cache)

    def _cleanup_tempdir(self):
        # Windows 下 Chroma 会占用文件，先 GC 再删；删不掉也不让测试失败
        self.doc_store = None
        self.store = None
        gc.collect()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _count(self) -> int:
        return self.store.get_collection_info()["count"]

    # ------------------------------------------------------------------
    # 一、构造
    # ------------------------------------------------------------------
    def test_构造时embedding_model为空抛ValueError(self):
        with self.assertRaises(ValueError):
            DocumentStore(embedding_model=None, vector_store=self.store)

    def test_构造时传入向量库则直接复用(self):
        self.assertIs(self.doc_store.vector_store, self.store)

    # ------------------------------------------------------------------
    # 二、add_parse_result
    # ------------------------------------------------------------------
    def test_add_parse_result_写入并生成确定性id(self):
        chunks = [
            {
                "chunk_id": 0,
                "type": "text",
                "content": "员工如何申请年假",
                "page": None,
                "metadata": {"file_name": "", "type_index": 0, "char_count": 8},
            },
            {
                "chunk_id": 1,
                "type": "table",
                "content": "| 假期 | 天数 |\n|---|---|\n| 年假 | 5 |",
                "page": 2,
                "metadata": {
                    "file_name": "hr.md",
                    "type_index": 0,
                    "char_count": 30,
                    "table_index": 0,
                    "cols": 2,
                    "row_count": 1,
                },
            },
        ]

        ids = self.doc_store.add_parse_result(
            make_parse_result(chunks), namespace="hr"
        )

        # id = namespace + file_name + chunk_id
        self.assertEqual(ids, ["hr_hr.md_0", "hr_hr.md_1"])
        self.assertEqual(self._count(), 2)
        # 整批只批量嵌入一次
        self.assertEqual(len(self.embedding.batch_calls), 1)
        self.assertEqual(
            self.embedding.batch_calls[0], [c["content"] for c in chunks]
        )

        stored = self.store.collection.get(
            ids=["hr_hr.md_0", "hr_hr.md_1"], include=["metadatas", "documents"]
        )
        text_meta, table_meta = stored["metadatas"]
        # 权威字段覆盖分块自带的值
        self.assertEqual(text_meta["file_name"], "hr.md")
        self.assertEqual(text_meta["chunk_type"], "text")
        self.assertEqual(text_meta["chunk_id"], 0)
        self.assertEqual(text_meta["char_count"], len(chunks[0]["content"]))
        self.assertEqual(text_meta["namespace"], "hr")
        self.assertNotIn("page", text_meta)
        # 表格块的页码与表格溯源字段都保留
        self.assertEqual(table_meta["chunk_type"], "table")
        self.assertEqual(table_meta["page"], 2)
        self.assertEqual(table_meta["table_index"], 0)
        self.assertEqual(table_meta["row_count"], 1)

    def test_add_parse_result_跳过空白与非dict分块(self):
        chunks = [
            "不是字典的分块",
            {"chunk_id": 0, "type": "text", "content": "   "},
            {"chunk_id": 2, "type": "text", "content": "有效内容"},
        ]

        with self.assertLogs("src.database.document_store", level="WARNING"):
            ids = self.doc_store.add_parse_result(make_parse_result(chunks))

        self.assertEqual(ids, ["default_hr.md_2"])
        self.assertEqual(self._count(), 1)

    def test_add_parse_result_空结果返回空列表且不写库(self):
        self.assertEqual(self.doc_store.add_parse_result(None), [])
        self.assertEqual(
            self.doc_store.add_parse_result(make_parse_result([])), []
        )
        self.assertEqual(self._count(), 0)
        self.assertEqual(self.embedding.batch_calls, [])

    def test_add_parse_result_重复导入抛DuplicateIDError(self):
        chunks = [{"chunk_id": 0, "type": "text", "content": "员工如何申请年假"}]
        self.doc_store.add_parse_result(make_parse_result(chunks))

        revised = [{"chunk_id": 0, "type": "text", "content": "改版后的年假政策"}]
        with self.assertRaises(DuplicateIDError) as ctx:
            self.doc_store.add_parse_result(make_parse_result(revised))

        # 提示先删除旧记录，避免用户困惑
        self.assertIn("delete_document", str(ctx.exception))
        self.assertEqual(self._count(), 1)
        # 拦截发生在向量化之前：只发生过首轮批量嵌入
        self.assertEqual(len(self.embedding.batch_calls), 1)
        # 旧内容没有被静默覆盖（Chroma 原生 add 会忽略重复 id）
        stored = self.store.collection.get(
            ids=["default_hr.md_0"], include=["documents"]
        )
        self.assertEqual(stored["documents"], ["员工如何申请年假"])

    # ------------------------------------------------------------------
    # 三、add_documents
    # ------------------------------------------------------------------
    def test_add_documents_不修改传入metadata并写入namespace(self):
        metadatas = [{"source": "wiki"}]

        ids = self.doc_store.add_documents(
            ["差旅报销标准"], metadatas, namespace="finance"
        )

        self.assertEqual(len(ids), 1)
        # id 形如 finance_<uuid4.hex>：前缀 + 32 位十六进制
        self.assertTrue(ids[0].startswith("finance_"))
        self.assertEqual(len(ids[0]), len("finance_") + 32)
        # 调用方传入的字典没被污染
        self.assertEqual(metadatas, [{"source": "wiki"}])

        stored = self.store.collection.get(ids=ids, include=["metadatas"])
        self.assertEqual(stored["metadatas"][0]["source"], "wiki")
        self.assertEqual(stored["metadatas"][0]["namespace"], "finance")

    def test_add_documents_metadata不足按空字典补且跳过空白文本(self):
        with self.assertLogs("src.database.document_store", level="WARNING"):
            ids = self.doc_store.add_documents(
                ["有效文本", "   "], [{"source": "a"}], namespace="misc"
            )

        self.assertEqual(len(ids), 1)
        stored = self.store.collection.get(ids=ids, include=["metadatas"])
        self.assertEqual(
            stored["metadatas"][0], {"source": "a", "namespace": "misc"}
        )

    def test_add_documents_空输入返回空列表(self):
        self.assertEqual(self.doc_store.add_documents([], []), [])
        self.assertEqual(self._count(), 0)

    # ------------------------------------------------------------------
    # 四、search
    # ------------------------------------------------------------------
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

    def test_search_不传namespace时返回全部且分数降序(self):
        self._add_hr_and_life_docs()

        results = self.doc_store.search("年假申请", top_k=5)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "hr_hr.md_0")
        self.assertEqual(results[0]["content"], "年假申请流程")
        self.assertGreater(results[0]["score"], results[1]["score"])
        # 原始 distance 一并返回，便于调用方自定义阈值
        self.assertIsNotNone(results[0]["distance"])
        self.assertEqual(results[0]["metadata"]["namespace"], "hr")
        self.assertEqual(
            set(results[0].keys()),
            {"id", "content", "score", "distance", "metadata"},
        )

    def test_search_按namespace过滤(self):
        self._add_hr_and_life_docs()

        hr_results = self.doc_store.search("年假申请", top_k=5, namespace="hr")
        life_results = self.doc_store.search("年假申请", top_k=5, namespace="life")

        self.assertEqual([r["id"] for r in hr_results], ["hr_hr.md_0"])
        self.assertEqual([r["id"] for r in life_results], ["life_life.md_0"])

    def test_search_目标namespace记录较少时会扩大抓取量(self):
        # 3 条 life 文档与查询高度相似，hr 文档相似度低；top_k=1 只取 hr
        self.doc_store.add_documents(
            ["年假申请年假申请年假申请", "年假申请年假申请", "年假申请年假申请年假"],
            [{}, {}, {}],
            namespace="life",
        )
        self.doc_store.add_parse_result(
            make_parse_result(
                [{"chunk_id": 0, "type": "text", "content": "员工手册"}],
                file_name="hr.md",
            ),
            namespace="hr",
        )

        results = self.doc_store.search("年假申请", top_k=1, namespace="hr")

        # 若只抓 top_k=1 条会全部命中 life 而漏掉 hr，这里断言 over-fetch 生效
        self.assertEqual([r["id"] for r in results], ["hr_hr.md_0"])

    def test_search_空集合返回空列表(self):
        self.assertEqual(self.doc_store.search("任意问题"), [])
        self.assertEqual(
            self.doc_store.search("任意问题", namespace="hr"), []
        )

    def test_search_top_k非正数或空query返回空列表(self):
        self._add_hr_and_life_docs()

        self.assertEqual(self.doc_store.search("年假申请", top_k=0), [])
        with self.assertLogs("src.database.document_store", level="WARNING"):
            self.assertEqual(self.doc_store.search("  "), [])

    # ------------------------------------------------------------------
    # 五、统计与维护
    # ------------------------------------------------------------------
    def test_get_stats_包含集合信息与模型信息(self):
        self._add_hr_and_life_docs()

        stats = self.doc_store.get_stats()

        self.assertEqual(stats["name"], "test_document_store")
        self.assertEqual(stats["count"], 2)
        self.assertEqual(stats["distance"], "cosine")
        self.assertEqual(stats["persist_directory"], self.tmpdir)
        self.assertEqual(stats["embedding_model"], "fake-embedding")
        self.assertEqual(stats["embedding_dimension"], 16)

    def test_delete_document与clear(self):
        self._add_hr_and_life_docs()

        self.doc_store.delete_document("hr_hr.md_0")
        self.assertEqual(self._count(), 1)

        self.assertTrue(self.doc_store.clear())
        self.assertEqual(self._count(), 0)
        # 清空后检索拿到空列表，而不是越界报错
        self.assertEqual(self.doc_store.search("年假申请"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)


