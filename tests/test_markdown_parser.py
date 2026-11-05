# -*- coding: utf-8 -*-
"""
test_markdown_parser.py —— MarkdownParser 的单元测试。

用标准库 unittest 编写，无需额外安装 pytest。
MarkdownParser 走 docling 的 markdown 后端（规则解析，离线可用），
在临时目录写入含表格的 .md 文件验证解析结果。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.parsers.base import BaseParser, ParseResult
from src.parsers.markdown_parser import MarkdownParser


MD_CONTENT = """# 标题

这是第一段内容，用于验证正文提取。

这是第二段内容。

| 名称 | 数量 | 备注 |
| --- | --- | --- |
| 苹果 | 3 | 新鲜 |
| 香蕉 | 5 | 进口 |
"""


class TestMarkdownParserSupportFileTypes(unittest.TestCase):
    """support_file_types 的行为。"""

    def setUp(self):
        self.parser = MarkdownParser()

    def test_包含小写md扩展名(self):
        exts = self.parser.support_file_types()
        self.assertIn('.md', exts)

    def test_包含大写MD扩展名(self):
        exts = self.parser.support_file_types()
        self.assertIn('.MD', exts)


class TestMarkdownParserParse(unittest.TestCase):
    """从文件路径解析 markdown 文件。"""

    def setUp(self):
        self.parser = MarkdownParser()
        self.tmpdir = tempfile.mkdtemp()
        self.md_path = os.path.join(self.tmpdir, "sample.md")
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(MD_CONTENT)

    def test_返回ParseResult对象(self):
        result = self.parser.parse(self.md_path)
        self.assertIsInstance(result, ParseResult)

    def test_raw_text非空(self):
        result = self.parser.parse(self.md_path)
        self.assertTrue(result.raw_text.strip())

    def test_文件名正确(self):
        result = self.parser.parse(self.md_path)
        self.assertEqual(result.file_name, "sample.md")

    def test_total_pages兜底为1(self):
        result = self.parser.parse(self.md_path)
        self.assertEqual(result.total_pages, 1)

    def test_table_count为1(self):
        result = self.parser.parse(self.md_path)
        self.assertEqual(result.table_count, 1)

    def test_tables_rows为二维列表(self):
        result = self.parser.parse(self.md_path)
        table = result.tables[0]
        self.assertIsInstance(table["rows"], list)
        self.assertTrue(all(isinstance(r, list) for r in table["rows"]))

    def test_表格内容正确(self):
        result = self.parser.parse(self.md_path)
        table = result.tables[0]
        # docling 会把首行当表头：优先匹配列名，其次匹配数据行
        self.assertTrue(table["rows"][0] == ["名称", "数量", "备注"] or
                        table["rows"][1] == ["苹果", "3", "新鲜"])

    def test_metadata包含parser和file_path(self):
        result = self.parser.parse(self.md_path)
        self.assertEqual(result.metadata["parser"], "docling")
        self.assertEqual(result.metadata["file_path"], self.md_path)

    def tearDown(self):
        if os.path.exists(self.md_path):
            os.remove(self.md_path)
        try:
            os.rmdir(self.tmpdir)
        except OSError:
            pass


class TestMarkdownParserParseFromBytes(unittest.TestCase):
    """从字节流解析 markdown 文件。"""

    def setUp(self):
        self.parser = MarkdownParser()

    def test_返回ParseResult对象(self):
        result = self.parser.parse_from_bytes(
            MD_CONTENT.encode("utf-8"), "sample.md"
        )
        self.assertIsInstance(result, ParseResult)

    def test_raw_text非空(self):
        result = self.parser.parse_from_bytes(
            MD_CONTENT.encode("utf-8"), "sample.md"
        )
        self.assertTrue(result.raw_text.strip())

    def test_文件名正确(self):
        result = self.parser.parse_from_bytes(
            MD_CONTENT.encode("utf-8"), "sample.md"
        )
        self.assertEqual(result.file_name, "sample.md")

    def test_table_count为1(self):
        result = self.parser.parse_from_bytes(
            MD_CONTENT.encode("utf-8"), "sample.md"
        )
        self.assertEqual(result.table_count, 1)

    def test_表格内容正确(self):
        result = self.parser.parse_from_bytes(
            MD_CONTENT.encode("utf-8"), "sample.md"
        )
        table = result.tables[0]
        # docling 导出行为：首行进 DataFrame 列名还是数据行，两种都要能对上
        self.assertTrue(table["rows"][0] == ["名称", "数量", "备注"] or
                        table["rows"][1] == ["苹果", "3", "新鲜"])

    def test_metadata包含parser和file_name(self):
        result = self.parser.parse_from_bytes(
            MD_CONTENT.encode("utf-8"), "sample.md"
        )
        self.assertEqual(result.metadata["parser"], "docling")
        self.assertEqual(result.metadata["file_path"], "sample.md")

    def test_不含表格时table_count为0(self):
        content = "只有正文，没有表格。"
        result = self.parser.parse_from_bytes(content.encode("utf-8"), "plain.md")
        self.assertEqual(result.table_count, 0)


if __name__ == "__main__":
    unittest.main()
