# -*- coding: utf-8 -*-
"""知识库管理接口测试：初始化失败返回可读 503、正常路径返回文件元数据（全部 mock）。

运行方式（项目根目录下）::

    .venv/Scripts/python.exe -m unittest tests.test_document_api -v
"""

import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routers import document_router


class FakeVectorStore:
    """只实现知识库接口用到的向量库方法，避免访问真实 Chroma。"""

    def __init__(self, files):
        self._files = list(files)

    def list_files(self, namespace=None):
        return [dict(item) for item in self._files]


class FakeEmbeddingModel:
    """假向量模型：只需提供模型名与维度。"""

    def get_model_name(self):
        return "fake-embedding"

    def get_dimension(self):
        return 1024


class FakeDocumentStore:
    """假文档存储：文件列表、统计与删除都返回固定数据。"""

    def __init__(self, files=None):
        self.vector_store = FakeVectorStore(files or [])
        self.embedding_model = FakeEmbeddingModel()

    def list_files(self, namespace=None):
        return self.vector_store.list_files(namespace=namespace)

    def get_stats(self):
        return {
            "name": "rag_documents",
            "count": 2,
            "distance": "cosine",
            "persist_directory": "fake",
        }

    def delete_by_file_name(self, file_name, namespace=None):
        return 2


class FakeRegistry:
    """假上传登记表：默认没有记录，便于断言接口回落到向量库元数据。"""

    def list_by_collection(self, collection):
        return []

    def count(self, collection=None):
        return 0

    def delete_by_name(self, collection, file_name):
        return True


class DocumentAPITestBase(unittest.TestCase):
    """统一准备：假集合存在性、假存储、假登记表。"""

    files = [
        {
            "file_name": "员工手册.pdf",
            "namespace": "rag_documents",
            "chunks": 2,
            "tables": 0,
            "pages": [1],
            "total_chars": 1200,
        }
    ]

    def setUp(self):
        self.client = TestClient(app)
        patches = [
            mock.patch.object(
                document_router, "_collection_exists", lambda name: True
            ),
            mock.patch.object(
                document_router,
                "get_document_store",
                lambda name: FakeDocumentStore(self.files),
            ),
            mock.patch.object(
                document_router, "get_upload_registry", lambda: FakeRegistry()
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)


class DocumentFilesAPITest(DocumentAPITestBase):
    """文件列表接口：正常元数据与初始化失败分支。"""

    def test_文件列表返回入库元数据(self):
        response = self.client.get(
            "/api/document/files", params={"collection_name": "rag_documents"}
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["collection_name"], "rag_documents")
        self.assertEqual(len(payload["files"]), 1)
        file_info = payload["files"][0]
        self.assertEqual(file_info["file_name"], "员工手册.pdf")
        self.assertEqual(file_info["chunks"], 2)
        # 登记表没有记录时回落到 success，前端展示为“已入库”
        self.assertEqual(file_info["status"], "success")

    def test_集合不存在返回空列表(self):
        with mock.patch.object(
            document_router, "_collection_exists", lambda name: False
        ):
            response = self.client.get(
                "/api/document/files", params={"collection_name": "rag_documents"}
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["files"], [])

    def test_存储初始化失败返回可读503(self):
        """配置/向量库异常必须翻译成 503 + 原因，而不是裸 500。"""
        error = ValueError(
            "distance 只支持 ('l2', 'ip', 'cosine')，"
            "当前传入 'cosine#向量相似度空间'"
        )
        with mock.patch.object(
            document_router, "get_document_store", side_effect=error
        ):
            response = self.client.get(
                "/api/document/files", params={"collection_name": "rag_documents"}
            )

        self.assertEqual(response.status_code, 503)
        detail = response.json()["detail"]
        self.assertIn("rag_documents", detail)
        self.assertIn("初始化失败", detail)
        self.assertIn("distance 只支持", detail)


class DocumentStatsAPITest(DocumentAPITestBase):
    """集合统计接口：正常返回与初始化失败分支。"""

    def test_统计接口返回向量数文件数与登记数(self):
        response = self.client.get(
            "/api/document/stats", params={"collection_name": "rag_documents"}
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["count"], 2)
        self.assertEqual(payload["files"], 1)
        self.assertEqual(payload["registry_records"], 0)
        self.assertEqual(payload["distance"], "cosine")

    def test_存储初始化失败返回可读503(self):
        with mock.patch.object(
            document_router, "get_document_store", side_effect=ValueError("配置非法")
        ):
            response = self.client.get(
                "/api/document/stats", params={"collection_name": "rag_documents"}
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("配置非法", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
