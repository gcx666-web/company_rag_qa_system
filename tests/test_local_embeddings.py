# -*- coding: utf-8 -*-
"""LocalEmbeddings 本地向量化模块的测试（Mock 单元测试 + 真实模型集成测试）。

运行方式（在项目根目录 D:/work/company_rag_qa_system 下）::

    .venv/Scripts/python.exe -m unittest tests.test_local_embeddings -v

也可以直接运行本文件::

    .venv/Scripts/python.exe tests/test_local_embeddings.py

说明：
1. Mock 部分用 unittest.mock 替换 SentenceTransformer，不加载模型、无网络依赖，
   秒级完成，验证构造参数、encode 调用参数、返回值类型转换等调用链。
2. 真实模型部分默认使用本机已缓存的 BAAI/bge-small-zh（512 维，中文小模型），
   可离线运行；如需换模型或设备，用环境变量覆盖：
       LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
       LOCAL_EMBEDDING_DEVICE=cuda:0
   模型无法加载（未缓存且无网络等）时会打印 [SKIP] 并跳过该组用例，
   不会让整套测试失败。
3. HuggingFace 镜像与 Xet 开关必须在 import sentence_transformers 之前设置，
   否则 hf-mirror.com 的 Xet 协议会返回 401、官方源可能连接超时
   （与 tests/test_pdf.py 的做法保持一致）。
4. 用标准库 unittest 编写，无需额外安装 pytest。
"""

import math
import os
import sys
import unittest
from unittest import mock

# 必须在导入 sentence_transformers 之前设置，避免走不可用的下载通道
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# 把项目根目录加入 sys.path，保证用任意方式、从任意目录运行都能导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.embeddings.local_embeddings import LocalEmbeddings


# 真实模型测试使用的模型与设备：默认用已缓存的中文小模型，可用环境变量覆盖
REAL_MODEL_NAME = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh")
REAL_DEVICE = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")

# 用于语义相似度检查的三条文本：前两条语义相近，第三条无关
TEXT_A = "员工如何申请年假"
TEXT_B = "请假需要走什么流程"
TEXT_C = "苹果手机是电子产品"


def cosine_similarity(left, right):
    """计算两个向量的余弦相似度（纯标准库实现，避免引入额外依赖）。"""
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


# ======================================================================
# 一、Mock 单元测试：只验证 LocalEmbeddings 自身的调用链
# ======================================================================
class TestLocalEmbeddingsMocked(unittest.TestCase):
    """用 mock 替换 SentenceTransformer，不加载真实模型。"""

    def setUp(self):
        # patch 的是模块内的名字，必须作用在 src.embeddings.local_embeddings 上
        patcher = mock.patch("src.embeddings.local_embeddings.SentenceTransformer")
        self.mock_sentence_transformer = patcher.start()
        self.addCleanup(patcher.stop)
        self.mock_model = self.mock_sentence_transformer.return_value

    def test_构造时把模型名与设备传给SentenceTransformer(self):
        embedder = LocalEmbeddings(model_name="fake-model", device="cuda:0")
        self.mock_sentence_transformer.assert_called_once_with(
            "fake-model", device="cuda:0"
        )
        self.assertEqual(embedder.model_name, "fake-model")
        self.assertEqual(embedder.device, "cuda:0")
        self.assertIs(embedder.model, self.mock_model)

    def test_默认模型名与设备(self):
        embedder = LocalEmbeddings()
        self.mock_sentence_transformer.assert_called_once_with(
            "all-MiniLM-L6-v2", device="cpu"
        )
        self.assertEqual(embedder.get_model_name(), "all-MiniLM-L6-v2")
        self.assertEqual(embedder.device, "cpu")

    def test_embed_text_转发参数并返回float列表(self):
        self.mock_model.encode.return_value = np.array(
            [0.1, -0.2, 0.3], dtype=np.float32
        )
        embedder = LocalEmbeddings(model_name="fake-model", device="cpu")
        vector = embedder.embed_text("一段文本")

        self.mock_model.encode.assert_called_once_with(
            "一段文本", convert_to_numpy=True, device="cpu"
        )
        self.assertIsInstance(vector, list)
        self.assertEqual(len(vector), 3)
        self.assertTrue(all(isinstance(value, float) for value in vector))
        for actual, expected in zip(vector, [0.1, -0.2, 0.3]):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_embed_batch_转发参数并返回二维列表(self):
        self.mock_model.encode.return_value = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32
        )
        embedder = LocalEmbeddings(model_name="fake-model", device="cpu")
        vectors = embedder.embed_batch(["文本一", "文本二"])

        self.mock_model.encode.assert_called_once_with(
            ["文本一", "文本二"], convert_to_numpy=True, device="cpu"
        )
        self.assertIsInstance(vectors, list)
        self.assertEqual(len(vectors), 2)
        for row in vectors:
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), 3)
            self.assertTrue(all(isinstance(value, float) for value in row))

    def test_embed_batch_空输入返回空列表(self):
        # sentence-transformers 对空列表返回空 ndarray，这里应转成 []
        self.mock_model.encode.return_value = np.array([], dtype=np.float64)
        embedder = LocalEmbeddings(model_name="fake-model", device="cpu")
        self.assertEqual(embedder.embed_batch([]), [])
        self.mock_model.encode.assert_called_once_with(
            [], convert_to_numpy=True, device="cpu"
        )

    def test_get_model_name_返回构造时的模型名(self):
        embedder = LocalEmbeddings(model_name="fake-model", device="cpu")
        self.assertEqual(embedder.get_model_name(), "fake-model")

    def test_get_dimension_返回模型维度(self):
        self.mock_model.get_embedding_dimension.return_value = 384
        embedder = LocalEmbeddings(model_name="fake-model", device="cpu")
        self.assertEqual(embedder.get_dimension(), 384)
        self.mock_model.get_embedding_dimension.assert_called_once_with()


# ======================================================================
# 二、真实模型集成测试：加载真实 SentenceTransformer，验证向量效果
# ======================================================================
class TestLocalEmbeddingsRealModel(unittest.TestCase):
    """真实加载本地模型，验证向量维度/数值/语义；模型不可用时整组跳过。"""

    embedder = None

    @classmethod
    def setUpClass(cls):
        try:
            cls.embedder = LocalEmbeddings(
                model_name=REAL_MODEL_NAME, device=REAL_DEVICE
            )
        except Exception as exc:  # 模型未缓存且无网络等环境问题不应让整套测试失败
            raise unittest.SkipTest(
                f"本地模型 {REAL_MODEL_NAME} 加载失败"
                f"（{type(exc).__name__}: {exc}）；"
                "请确认该模型已缓存或网络可用，"
                "也可用 LOCAL_EMBEDDING_MODEL 指定其他本地模型"
            )

    def test_单条文本向量维度与数值有效(self):
        vector = self.embedder.embed_text(TEXT_A)
        model_dimension = self.embedder.model.get_embedding_dimension()

        self.assertIsInstance(vector, list)
        self.assertTrue(vector, "单条文本返回了空向量")
        self.assertEqual(len(vector), model_dimension)
        self.assertEqual(self.embedder.get_dimension(), model_dimension)
        self.assertTrue(
            all(math.isfinite(value) for value in vector),
            "向量中存在 NaN/Inf",
        )

    def test_批量向量条数与维度一致(self):
        texts = [TEXT_A, TEXT_B, TEXT_C]
        vectors = self.embedder.embed_batch(texts)

        self.assertEqual(len(vectors), len(texts))
        dimensions = {len(vector) for vector in vectors}
        self.assertEqual(dimensions, {self.embedder.get_dimension()})

    def test_批量结果与单条结果一致(self):
        batch_vectors = self.embedder.embed_batch([TEXT_A, TEXT_B])
        single_vector = self.embedder.embed_text(TEXT_A)

        # 批处理与单条走的是同一个 encode，结果应几乎完全一致
        self.assertGreater(
            cosine_similarity(batch_vectors[0], single_vector), 0.9999
        )

    def test_同义句相似度高于无关句(self):
        vectors = self.embedder.embed_batch([TEXT_A, TEXT_B, TEXT_C])
        sim_related = cosine_similarity(vectors[0], vectors[1])
        sim_unrelated = cosine_similarity(vectors[0], vectors[2])

        # 只比较相对高低，不写死阈值，避免不同模型尺度差异导致用例脆弱
        print(f"\n  {TEXT_A} <-> {TEXT_B}: {sim_related:.6f}")
        print(f"  {TEXT_A} <-> {TEXT_C}: {sim_unrelated:.6f}")
        self.assertGreater(
            sim_related,
            sim_unrelated,
            "同义句的相似度应高于无关句，"
            f"当前 {sim_related:.6f} <= {sim_unrelated:.6f}",
        )

    def test_空批量返回空列表(self):
        self.assertEqual(self.embedder.embed_batch([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

