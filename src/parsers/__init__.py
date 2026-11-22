# -*- coding: utf-8 -*-
"""解析器工厂。

优化点：每个解析器在构造时都会创建 Docling ``DocumentConverter``（包含 OCR /
表格结构模型配置），开销不小。这里按“线程 + 扩展名”缓存解析器实例：

- 同一个 FastAPI 线程池线程处理多个上传时，直接复用已初始化的转换器；
- 使用 ``threading.local`` 而不是全局缓存，避免 Docling 转换器被多个线程
  同时调用产生竞争（Docling 官方未承诺 DocumentConverter 线程安全）。

如需强制使用全新实例（例如测试或修改了解析器配置），传 ``use_cache=False``。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List

from src.parsers.base import BaseParser, ParseResult
from src.parsers.pdf_parser import PdfParser
from src.parsers.word_parser import WordParser
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.htmlparser import HtmlParser


class ParserFactory:
    """
    解析器工厂类
    """

    _parsers = {
        '.pdf': PdfParser,
        '.docx': WordParser,
        '.md': MarkdownParser,
        '.html': HtmlParser,
        '.htm': HtmlParser,
    }

    # 线程本地缓存：threading.local() 会在每个线程里创建独立字典
    _local = threading.local()

    @classmethod
    def get_parser(cls, file_path: str, use_cache: bool = True) -> BaseParser:
        """
        根据文件路径获取对应的解析器。

        直接按扩展名查表，不再遍历所有解析器类；原来为了比对后缀会把每个
        解析器都实例化一次，而每个解析器构造时都会创建 Docling
        ``DocumentConverter``，开销很大。

        :param file_path: 文件路径或文件名。
        :param use_cache: 是否复用当前线程已创建的解析器实例（默认 True）。
        """
        ext = Path(file_path).suffix.lower()
        parser_cls = cls._parsers.get(ext)
        if parser_cls is None:
            raise ValueError(f"不支持的文件类型: {ext}")

        if not use_cache:
            return parser_cls()

        cache = getattr(cls._local, "instances", None)
        if cache is None:
            cache = {}
            cls._local.instances = cache

        parser = cache.get(ext)
        if parser is None:
            parser = parser_cls()
            cache[ext] = parser
        return parser

    @classmethod
    def clear_thread_cache(cls) -> None:
        """清空当前线程的解析器缓存（测试或释放 Docling 资源时使用）。"""
        cls._local.instances = {}

    @classmethod
    def register_parser(cls, ext: str, parser_cls: BaseParser):
        """
        注册新的解析器

        args:
            ext: 文件扩展名
            parser_cls: 继承自 BaseParser 的解析器类
        """
        cls._parsers[ext.lower()] = parser_cls
        # 注册新解析器后清空当前线程缓存，避免仍返回旧的空缓存判断
        cls.clear_thread_cache()

    @classmethod
    def support_file_types(cls) -> List[str]:
        """
        返回所有支持的文件扩展名
        """
        return list(cls._parsers.keys())


# 调用函数
def parse_file(file_path: str) -> ParseResult:
    """
    调用函数
    根据文件路径解析文档
    """
    parser = ParserFactory.get_parser(file_path)
    return parser.parse(file_path)
