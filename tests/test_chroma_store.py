# -*- coding: utf-8 -*-
"""ChromaVectorStore 向量存储模块的测试（Mock 单元测试 + 真实 Chroma 集成测试）。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    .venv/Scripts/python.exe -m unittest tests.test_chroma_store -v

也可以直接运行本文件::

    .venv/Scripts/python.exe tests/test_chroma_store.py

说明：
1. Mock 部分把 chromadb.PersistentClient 替换成 MagicMock，不落盘、无网络依赖，
   验证入参归一化、数量校验、重复 id 提示等调用链；
2. 集成部分使用临时目录创建真实的 PersistentClient（向量由调用方显式传入，
   不会触发默认 embedding 模型的下载），验证写入/检索/删除/清空的实际行为；
3. Windows 下 Chroma 会占用 sqlite/hnsw 文件，tearDown 先清 Chroma 的系统缓存
   再删临时目录，删除失败不视为测试失败（rmtree(ignore_errors=True)）。
4. 用标准库 unittest 编写，无需额外安装 pytest。
"""

import gc
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

# 把项目根目录加入 sys.path，保证用任意方式、从任意目录运行都能导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chromadb.errors import DuplicateIDError

from src.embeddings.chorm_store import ChromaVectorStore
from src.embeddings.vector_store import Document


def make_document(
    document_id="doc-1",
    content="测试内容",
    metadata=None,
) -> Document:
    """构造测试用 Document；metadata 默认给一个最小合法字典。"""
    return Document(
        id=document_id,
        content=content,
        metadata={"file_name": "demo.md"} if metadata is None else metadata,
    )


# ======================================================================
# 一、Mock 单元测试：只验证 ChromaVectorStore 自身的调用链
# ======================================================================
class TestChromaVectorStoreMocked(unittest.TestCase):
    """用 mock 替换 PersistentClient，不创建真实数据库。"""

    def setUp(self):
        patcher = mock.patch("src.embeddings.chorm_store.chromadb.PersistentClient")
        self.mock_persistent_client = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_client = self.mock_persistent_client.return_value
        self.mock_collection = self.mock_client.get_or_create_collection.return_value

    def test_构造时按参数创建持久化客户端与集合(self):
        store = ChromaVectorStore(
            collection_name="my_docs",
            persist_directory="./tmp/chroma",
            distance="l2",
        )

        self.mock_persistent_client.assert_called_once_with(path="./tmp/chroma")
        self.mock_client.get_or_create_collection.assert_called_once_with(
            name="my_docs", metadata={"hnsw:space": "l2"}
        )
        self.assertEqual(store.collection_name, "my_docs")
        self.assertEqual(store.persist_directory, "./tmp/chroma")
        # 距离必须是字符串（写成 distance, 会变成元组，Chroma 直接报错）
        self.assertEqual(store.distance, "l2")
        self.assertIsInstance(store.distance, str)

    def test_distance非法时抛ValueError(self):
        with self.assertRaises(ValueError):
            ChromaVectorStore(distance="cosin")

    def test_add_documents_空输入返回空列表且不写库(self):
        store = ChromaVectorStore()
        self.assertEqual(store.add_documents([], []), [])
        self.mock_collection.add.assert_not_called()

    def test_add_documents_数量不一致抛ValueError(self):
        store = ChromaVectorStore()
        with self.assertRaises(ValueError):
            store.add_documents(
                [make_document("a"), make_document("b")], [[0.1, 0.2]]
            )
        self.mock_collection.add.assert_not_called()

    def test_add_documents_整型id转成字符串并回写文档(self):
        store = ChromaVectorStore()
        document = make_document(1, "正文", {"file_name": "a.md"})

        ids = store.add_documents([document], [[0.1, 0.2, 0.3]])

        self.assertEqual(ids, ["1"])
        self.assertEqual(document.id, "1")
        self.mock_collection.add.assert_called_once_with(
            ids=["1"],
            documents=["正文"],
            embeddings=[[0.1, 0.2, 0.3]],
            metadatas=[{"file_name": "a.md"}],
        )

    def test_add_documents_空id自动生成uuid并回写(self):
        store = ChromaVectorStore()
        document = make_document("", "无 id 的正文")

        ids = store.add_documents([document], [[0.1, 0.2, 0.3]])

        self.assertEqual(len(ids), 1)
        # uuid4 的字符串形式固定为 36 位
        self.assertEqual(len(ids[0]), 36)
        self.assertEqual(document.id, ids[0])

    def test_add_documents_metadata只保留Chroma支持的标量(self):
        store = ChromaVectorStore()
        document = make_document(
            "doc-1",
            "正文",
            {
                "file_name": "a.md",   # str 保留
                "char_count": 12,      # int 保留
                "score": 0.5,          # float 保留
                "flag": True,          # bool 保留
                "cols": None,          # None 丢弃
                "nested": {"a": 1},    # 嵌套 dict 丢弃
                "tags": ["x", "y"],    # list 丢弃
            },
        )

        with self.assertLogs("src.embeddings.chorm_store", level="WARNING"):
            store.add_documents([document], [[0.1, 0.2, 0.3]])

        metadatas = self.mock_collection.add.call_args.kwargs["metadatas"]
        self.assertEqual(
            metadatas,
            [{"file_name": "a.md", "char_count": 12, "score": 0.5, "flag": True}],
        )


    def test_add_documents_metadata为空时写入None(self):
        store = ChromaVectorStore()
        document = make_document("doc-1", "正文", {})

        store.add_documents([document], [[0.1, 0.2, 0.3]])

        metadatas = self.mock_collection.add.call_args.kwargs["metadatas"]
        self.assertEqual(metadatas, [None])

    def test_add_documents_同批重复id抛ValueError(self):
        store = ChromaVectorStore()
        with self.assertRaises(ValueError):
            store.add_documents(
                [make_document("same"), make_document("same")],
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            )
        self.mock_collection.add.assert_not_called()

    def test_add_documents_集合已有同id时提示先删除(self):
        store = ChromaVectorStore()
        self.mock_collection.add.side_effect = DuplicateIDError(
            "Expected IDs to be unique, found duplicates of: doc-1"
        )

        with self.assertRaises(DuplicateIDError) as ctx:
            store.add_documents([make_document("doc-1")], [[0.1, 0.2, 0.3]])

        self.assertIn("delete_document", str(ctx.exception))

    def test_search_空集合返回空列表(self):
        store = ChromaVectorStore()
        self.mock_collection.count.return_value = 0

        self.assertEqual(store.search([0.1, 0.2, 0.3], top_k=5), [])
        self.mock_collection.query.assert_not_called()

    def test_search_top_k非正数返回空列表(self):
        store = ChromaVectorStore()
        self.assertEqual(store.search([0.1, 0.2, 0.3], top_k=0), [])

    def test_search_空查询向量抛ValueError(self):
        store = ChromaVectorStore()
        self.mock_collection.count.return_value = 1
        with self.assertRaises(ValueError):
            store.search([], top_k=1)

    def test_search_限制n_results并换算成相似度分数(self):
        store = ChromaVectorStore(distance="cosine")
        self.mock_collection.count.return_value = 3
        self.mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "documents": [["员工如何申请年假", "请假需要走什么流程"]],
            "metadatas": [[{"file_name": "a.md"}, None]],
            "distances": [[0.1, 0.4]],
        }

        results = store.search([1.0, 0.0, 0.0], top_k=10)

        # n_results 不能超过集合条数，否则 Chroma 会报越界
        self.mock_collection.query.assert_called_once_with(
            query_embeddings=[[1.0, 0.0, 0.0]],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )
        self.assertEqual([r.document.id for r in results], ["1", "2"])
        self.assertEqual(results[0].document.content, "员工如何申请年假")
        self.assertAlmostEqual(results[0].score, 0.9)
        self.assertAlmostEqual(results[1].score, 0.6)
        # 原始 distance 保留在 SearchResult.metadata 里，便于自定义阈值
        self.assertAlmostEqual(results[0].metadata["distance"], 0.1)
        self.assertEqual(results[0].metadata["file_name"], "a.md")
        self.assertEqual(results[1].document.metadata, {})

    def test_search_l2空间按距离映射分数(self):
        store = ChromaVectorStore(distance="l2")
        # l2 空间下 score = 1 / (1 + distance)
        self.assertAlmostEqual(store._distance_to_score(3.0), 0.25)
        self.assertAlmostEqual(store._distance_to_score(None), 0.0)


    def test_delete_document_按字符串id删除(self):
        store = ChromaVectorStore()
        store.delete_document(123)
        self.mock_collection.delete.assert_called_once_with(ids=["123"])

    def test_delete_document_空id静默跳过(self):
        store = ChromaVectorStore()
        with self.assertLogs("src.embeddings.chorm_store", level="WARNING"):
            store.delete_document("")
        self.mock_collection.delete.assert_not_called()

    def test_clear_删除集合并按配置重建(self):
        store = ChromaVectorStore(collection_name="docs", distance="ip")
        new_collection = mock.MagicMock()
        # clear() 会再次调用 get_or_create_collection，这里让它返回新集合
        self.mock_client.get_or_create_collection.return_value = new_collection

        self.assertTrue(store.clear())

        self.mock_client.delete_collection.assert_called_once_with(name="docs")
        self.assertIs(store.collection, new_collection)

    def test_clear_删除集合失败返回False(self):
        store = ChromaVectorStore()
        self.mock_client.delete_collection.side_effect = RuntimeError("boom")

        with self.assertLogs("src.embeddings.chorm_store", level="ERROR"):
            self.assertFalse(store.clear())

    def test_get_collection_info_返回名称条数距离与目录(self):
        store = ChromaVectorStore(collection_name="docs", persist_directory="./db")
        self.mock_collection.count.return_value = 7

        self.assertEqual(
            store.get_collection_info(),
            {
                "name": "docs",
                "count": 7,
                "distance": "cosine",
                "persist_directory": "./db",
            },
        )


# ======================================================================
# 二、真实 Chroma 集成测试：临时目录 + 显式向量，不下载模型
# ======================================================================
class TestChromaVectorStoreIntegration(unittest.TestCase):
    """使用真实 PersistentClient（临时目录），验证完整读写链路。"""

    def setUp(self):
        from chromadb.api.shared_system_client import SharedSystemClient

        self.addCleanup(self._cleanup_tempdir)
        self.tmpdir = tempfile.mkdtemp(prefix="chroma_store_test_")
        self.store = ChromaVectorStore(
            collection_name="test_documents", persist_directory=self.tmpdir
        )
        self.addCleanup(SharedSystemClient.clear_system_cache)

    def _cleanup_tempdir(self):
        # Windows 下 Chroma 会占用文件，先 GC 再删；删不掉也不让测试失败
        self.store = None
        gc.collect()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_写入后条目数增加且metadata已归一化(self):
        documents = [
            make_document(
                "chunk-0",
                "员工如何申请年假",
                {"file_name": "hr.md", "char_count": 7, "cols": None},
            ),
            make_document("chunk-1", "请假需要走什么流程", {"file_name": "hr.md"}),
        ]

        ids = self.store.add_documents(
            documents, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )

        self.assertEqual(ids, ["chunk-0", "chunk-1"])
        self.assertEqual(self.store.get_collection_info()["count"], 2)

        stored = self.store.collection.get(ids=["chunk-0"], include=["metadatas"])
        self.assertEqual(
            stored["metadatas"][0], {"file_name": "hr.md", "char_count": 7}
        )

    def test_检索返回最相似文档且分数降序(self):
        self.store.add_documents(
            [
                make_document("a", "年假申请流程", {"file_name": "hr.md"}),
                make_document("b", "食堂菜单介绍", {"file_name": "life.md"}),
            ],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        results = self.store.search([1.0, 0.0, 0.0], top_k=5)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].document.id, "a")
        self.assertEqual(results[0].document.content, "年假申请流程")
        self.assertGreater(results[0].score, results[1].score)
        self.assertIn("distance", results[0].metadata)

    def test_删除与清空后条目数正确(self):
        self.store.add_documents(
            [make_document("a"), make_document("b")],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        self.store.delete_document("a")
        self.assertEqual(self.store.get_collection_info()["count"], 1)

        self.assertTrue(self.store.clear())
        self.assertEqual(self.store.get_collection_info()["count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

