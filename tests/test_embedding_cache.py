# -*- coding: utf-8 -*-
"""EmbeddingCache / CachedEmbeddings 单元测试（纯内存 + 临时 SQLite，无网络）。

运行方式::

    .venv/Scripts/python.exe -m unittest tests.test_embedding_cache -v
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.embeddings.base import Base
from src.embeddings.embedding_cache import (
    CachedEmbeddings,
    EmbeddingCache,
)


class CountingEmbedding(Base):
    """按文本长度生成确定性向量，并记录真实调用次数。"""

    def __init__(self, model_name="counting-model", dimension=4):
        self.model_name = model_name
        self.dimension = dimension
        self.calls = []

    def embed_text(self, text):
        return self.embed_batch([text])[0]

    def embed_batch(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            base = float(len(str(text)))
            vectors.append([base + index for index in range(self.dimension)])
        return vectors

    def get_dimension(self):
        return self.dimension

    def get_model_name(self):
        return self.model_name


class TestEmbeddingCacheMemory(unittest.TestCase):
    def test_命中内存后不再读取磁盘(self):
        cache = EmbeddingCache(db_path=None, max_memory_items=10)
        self.assertIsNone(cache.get("m", "hello"))

        cache.put_many("m", ["hello"], [[0.1, 0.2]])
        self.assertEqual(cache.get("m", "hello"), [0.1, 0.2])

        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["memory_items"], 1)

    def test_不同模型不共享缓存(self):
        cache = EmbeddingCache(db_path=None)
        cache.put_many("model-a", ["same"], [[1.0]])
        self.assertIsNone(cache.get("model-b", "same"))

    def test_LRU淘汰最久未使用(self):
        cache = EmbeddingCache(db_path=None, max_memory_items=2)
        cache.put_many("m", ["a"], [[1.0]])
        cache.put_many("m", ["b"], [[2.0]])
        cache.get("m", "a")  # a 变成最近使用
        cache.put_many("m", ["c"], [[3.0]])  # 应淘汰 b
        self.assertIsNotNone(cache.get("m", "a"))
        self.assertIsNotNone(cache.get("m", "c"))
        self.assertIsNone(cache.get("m", "b"))

    def test_关闭开关后不读写(self):
        cache = EmbeddingCache(db_path=None, enabled=False)
        cache.put_many("m", ["x"], [[1.0]])
        self.assertIsNone(cache.get("m", "x"))
        self.assertEqual(cache.stats()["hits"], 0)


class TestEmbeddingCacheDisk(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="embedding_cache_test_")
        self.db_path = os.path.join(self.tmpdir, "cache.sqlite3")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)

    def test_写入磁盘后新实例仍能命中(self):
        first = EmbeddingCache(db_path=self.db_path)
        first.put_many("m", ["持久化文本"], [[0.5, 0.6, 0.7]])
        first.close()

        second = EmbeddingCache(db_path=self.db_path)
        vector = second.get("m", "持久化文本")
        # 磁盘使用 float32 存储以节省空间，比较时允许精度误差
        self.assertIsNotNone(vector)
        self.assertEqual(len(vector), 3)
        for actual, expected in zip(vector, [0.5, 0.6, 0.7]):
            self.assertAlmostEqual(actual, expected, places=5)
        stats = second.stats()
        self.assertEqual(stats["disk_entries"], 1)
        self.assertEqual(stats["disk_hits"], 1)
        second.close()

    def test_清空缓存同时清理磁盘(self):
        cache = EmbeddingCache(db_path=self.db_path)
        cache.put_many("m", ["a", "b"], [[1.0], [2.0]])
        stats = cache.clear()
        self.assertEqual(stats["disk_entries"], 0)
        self.assertIsNone(cache.get("m", "a"))
        cache.close()

    def test_向量维度损坏时忽略该条(self):
        cache = EmbeddingCache(db_path=self.db_path)
        cache.put_many("m", ["x"], [[1.0, 2.0]])
        # 人为把 dimension 改错，模拟旧版本/手工损坏数据
        with cache._lock:
            cache._conn.execute("UPDATE embeddings SET dimension = 99")
            cache._conn.commit()
        cache.close()

        # 换一个新实例，避开内存缓存，确保读取的是被改坏的磁盘记录
        reader = EmbeddingCache(db_path=self.db_path)
        self.assertIsNone(reader.get("m", "x"))
        reader.close()


class TestCachedEmbeddings(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="cached_embeddings_test_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.cache = EmbeddingCache(
            db_path=os.path.join(self.tmpdir, "cache.sqlite3"), max_memory_items=100
        )
        self.addCleanup(self.cache.close)
        self.inner = CountingEmbedding()
        self.model = CachedEmbeddings(
            inner=self.inner, cache=self.cache, cache_name="test", batch_size=2
        )

    def test_重复文本只调用一次真实模型(self):
        first = self.model.embed_batch(["a", "bb"])
        second = self.model.embed_batch(["a", "bb"])

        self.assertEqual(first, second)
        self.assertEqual(len(self.inner.calls), 1)
        self.assertEqual(self.inner.calls[0], ["a", "bb"])
        self.assertEqual(self.cache.stats()["hits"], 2)

    def test_只对未命中文本调用真实模型且顺序不变(self):
        self.model.embed_batch(["a"])
        vectors = self.model.embed_batch(["bb", "a", "ccc"])

        self.assertEqual(vectors[1], [1.0, 2.0, 3.0, 4.0])
        # 第二次真实调用只应包含 bb / ccc
        self.assertEqual(self.inner.calls[-1], ["bb", "ccc"])
        self.assertEqual(len(vectors), 3)

    def test_单条与批量结果一致(self):
        batch = self.model.embed_batch(["x", "y"])
        single = self.model.embed_text("x")
        self.assertEqual(single, batch[0])

    def test_空输入不调用模型(self):
        self.assertEqual(self.model.embed_batch([]), [])
        self.assertEqual(self.inner.calls, [])

    def test_透传模型名与维度(self):
        self.assertEqual(self.model.get_model_name(), "counting-model")
        self.assertEqual(self.model.get_dimension(), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
