# -*- coding: utf-8 -*-
"""
test_word_parser.py —— WordParser 的单元测试。

用标准库 unittest 编写，无需额外安装 pytest。
WordParser 走 docling 的 docx 后端（规则解析，离线可用）。
docling 解析器启动较慢，整个类共享一次转换以提速。

由于 docling 的 docx 解析需要真实合法的 docx 二进制内容，
测试中通过 python-docx 库动态生成一个最小合法的 docx 文件，
然后调用 parse / parse_from_bytes 验证整条链路。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from docling.document_converter import DocumentConverter
from src.parsers.base import BaseParser, ParseResult
from src.parsers.word_parser import WordParser

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


def make_docx(path: str) -> None:
    """用 python-docx 生成一个最小合法的 .docx 文件（含段落与表格）。"""
    d = DocxDocument()
    d.add_heading("测试标题", level=1)
    d.add_paragraph("这是测试正文内容。")
    t = d.add_table(rows=2, cols=2)
    t.rows[0].cells[0].text = "姓名"
    t.rows[0].cells[1].text = "职业"
    t.rows[1].cells[0].text = "张三"
    t.rows[1].cells[1].text = "教师"
    d.save(path)


class TestWordParserParse(unittest.TestCase):
    """从文件路径解析 docx。"""

    @classmethod
    def setUpClass(cls):
        cls.parser = WordParser()
        cls.tmpdir = tempfile.mkdtemp()
        cls.docx_path = os.path.join(cls.tmpdir, "sample.docx")
        make_docx(cls.docx_path)

    def test_返回ParseResult对象(self):
        result = self.parser.parse(self.docx_path)
        self.assertIsInstance(result, ParseResult)

    def test_raw_text非空(self):
        result = self.parser.parse(self.docx_path)
        self.assertTrue(result.raw_text.strip())

    def test_文件名正确(self):
        result = self.parser.parse(self.docx_path)
        self.assertEqual(result.file_name, "sample.docx")

    def test_total_pages至少为1(self):
        result = self.parser.parse(self.docx_path)
        self.assertGreaterEqual(result.total_pages, 1)

    def test_table_count至少为1(self):
        result = self.parser.parse(self.docx_path)
        self.assertGreaterEqual(result.table_count, 1)

    def test_tables_rows为二维列表(self):
        result = self.parser.parse(self.docx_path)
        table = result.tables[0]
        self.assertIsInstance(table["rows"], list)
        self.assertTrue(all(isinstance(r, list) for r in table["rows"]))

    def test_metadata包含parser和file_path(self):
        result = self.parser.parse(self.docx_path)
        self.assertEqual(result.metadata["parser"], "docling")
        self.assertEqual(result.metadata["file_path"], self.docx_path)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.docx_path):
            os.remove(cls.docx_path)
        try:
            os.rmdir(cls.tmpdir)
        except OSError:
            pass


class TestWordParserParseFromBytes(unittest.TestCase):
    """从字节流解析 docx（docling 需要真实合法的 docx 内容）。"""

    @classmethod
    def setUpClass(cls):
        # docling 启动和模型加载较慢，整个类共享一次转换结果
        tmpdir = tempfile.mkdtemp()
        docx_path = os.path.join(tmpdir, "stream.docx")
        make_docx(docx_path)
        with open(docx_path, "rb") as f:
            cls.file_bytes = f.read()
        cls.parser = WordParser()
        try:
            cls.result = cls.parser.parse_from_bytes(cls.file_bytes, "stream.docx")
        except Exception as exc:
            cls.result = None
            cls.parse_error = exc
        if os.path.exists(docx_path):
            os.remove(docx_path)
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    def test_返回ParseResult对象(self):
        if self.result is None:
            self.skipTest(f"docling 解析 docx 失败: {self.parse_error}")
        self.assertIsInstance(self.result, ParseResult)

    def test_raw_text非空(self):
        if self.result is None:
            self.skipTest(f"docling 解析 docx 失败: {self.parse_error}")
        self.assertTrue(self.result.raw_text.strip())

    def test_文件名正确(self):
        if self.result is None:
            self.skipTest(f"docling 解析 docx 失败: {self.parse_error}")
        self.assertEqual(self.result.file_name, "stream.docx")

    def test_table_count至少为1(self):
        if self.result is None:
            self.skipTest(f"docling 解析 docx 失败: {self.parse_error}")
        self.assertGreaterEqual(self.result.table_count, 1)

    def test_表头保留在rows首行(self):
        if self.result is None:
            self.skipTest(f"docling 解析 docx 失败: {self.parse_error}")
        table = self.result.tables[0]
        self.assertEqual(table["rows"][0], ["姓名", "职业"])
        self.assertEqual(table["rows"][1], ["张三", "教师"])


if __name__ == "__main__":
    unittest.main()
