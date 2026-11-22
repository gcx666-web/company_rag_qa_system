# -*- coding: utf-8 -*-
"""UploadRegistry / UploadLockPool 单元测试（临时 SQLite，无网络）。"""

import os
import shutil
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.upload_registry import UploadLockPool, UploadRecord, UploadRegistry


class TestUploadRegistry(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="upload_registry_test_")
        self.addCleanup(shutil.rmtree, self.tmpdir, True)
        self.registry = UploadRegistry(os.path.join(self.tmpdir, "registry.sqlite3"))
        self.addCleanup(self.registry.close)

    def test_写入后可按名称与指纹查询(self):
        record = UploadRecord(
            collection="hr",
            file_name="a.md",
            sha256="abc",
            size=10,
            pages=1,
            chunks=3,
            tables=0,
        )
        self.assertTrue(self.registry.upsert(record))

        by_name = self.registry.get_by_name("hr", "a.md")
        by_hash = self.registry.get_by_hash("hr", "abc")
        self.assertIsNotNone(by_name)
        self.assertEqual(by_name.chunks, 3)
        self.assertIsNotNone(by_hash)
        self.assertEqual(by_hash.file_name, "a.md")

    def test_同名覆盖更新但保留创建时间(self):
        first = UploadRecord(
            collection="hr", file_name="a.md", sha256="v1", chunks=1
        )
        self.registry.upsert(first)
        created_at = self.registry.get_by_name("hr", "a.md").created_at

        time.sleep(0.01)
        second = UploadRecord(
            collection="hr", file_name="a.md", sha256="v2", chunks=2
        )
        self.registry.upsert(second)

        updated = self.registry.get_by_name("hr", "a.md")
        self.assertEqual(updated.sha256, "v2")
        self.assertEqual(updated.chunks, 2)
        self.assertAlmostEqual(updated.created_at, created_at, places=3)
        self.assertGreaterEqual(updated.updated_at, created_at)

    def test_处理中与失败状态(self):
        self.registry.mark_processing("hr", "a.md", "abc", 10)
        self.assertEqual(
            self.registry.get_by_name("hr", "a.md").status, "processing"
        )

        self.registry.mark_failed("hr", "a.md", "abc", 10, "boom")
        failed = self.registry.get_by_name("hr", "a.md")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.error, "boom")

    def test_删除单条与整个集合(self):
        self.registry.upsert(
            UploadRecord(collection="hr", file_name="a.md", sha256="a")
        )
        self.registry.upsert(
            UploadRecord(collection="hr", file_name="b.md", sha256="b")
        )
        self.registry.upsert(
            UploadRecord(collection="life", file_name="c.md", sha256="c")
        )

        self.assertTrue(self.registry.delete_by_name("hr", "a.md"))
        self.assertIsNone(self.registry.get_by_name("hr", "a.md"))
        self.assertEqual(self.registry.count("hr"), 1)

        self.assertTrue(self.registry.delete_by_collection("hr"))
        self.assertEqual(self.registry.count("hr"), 0)
        self.assertEqual(self.registry.count("life"), 1)

    def test_关闭去重后查询恒为空(self):
        disabled = UploadRegistry(
            os.path.join(self.tmpdir, "disabled.sqlite3"), enabled=False
        )
        disabled.upsert(UploadRecord(collection="hr", file_name="a", sha256="x"))
        self.assertIsNone(disabled.get_by_name("hr", "a"))
        self.assertEqual(disabled.count(), 0)


class TestUploadLockPool(unittest.TestCase):
    def test_同key互斥(self):
        pool = UploadLockPool()
        order = []

        def worker(name):
            with pool.hold("same-key"):
                order.append(f"{name}-enter")
                time.sleep(0.03)
                order.append(f"{name}-exit")

        threads = [
            threading.Thread(target=worker, args=(name,)) for name in ("a", "b")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # 同一把锁下，进入/退出必须成对出现，不能交错
        self.assertEqual(len(order), 4)
        self.assertTrue(order[0].endswith("-enter"))
        self.assertTrue(order[1].endswith("-exit"))
        self.assertEqual(order[0].split("-")[0], order[1].split("-")[0])

    def test_不同key可并行(self):
        pool = UploadLockPool()
        barrier = threading.Barrier(2)
        results = []

        def worker(key):
            with pool.hold(key):
                barrier.wait(timeout=2)
                results.append(key)

        threads = [
            threading.Thread(target=worker, args=(key,)) for key in ("k1", "k2")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(sorted(results), ["k1", "k2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
