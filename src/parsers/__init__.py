from pathlib import Path
from typing import List

from src.parsers.base import BaseParser, ParseResult
from src.parsers.pdf_parser import PdfParser
from src.parsers.word_parser import WordParser
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.htmlparser import HtmlParser


class ParserFactory:
    '''
    解析器工厂类    
    '''
    _parsers = {
        '.pdf': PdfParser,
        '.docx': WordParser,
        '.md': MarkdownParser,
        '.html': HtmlParser,
        '.htm': HtmlParser,
    }
    
    @classmethod
    def get_parser(cls, file_path: str) -> BaseParser:
        '''
        根据文件路径获取对应的解析器

        直接按扩展名查表，不再遍历所有解析器类；
        原来为了比对后缀会把每个解析器都实例化一次，
        而每个解析器构造时都会创建 Docling DocumentConverter，开销很大。
        '''
        ext = Path(file_path).suffix.lower()
        parser_cls = cls._parsers.get(ext)
        if parser_cls is None:
            raise ValueError(f"不支持的文件类型: {ext}")
        return parser_cls()
    
    @classmethod
    def register_parser(cls,ext:str ,parser_cls: BaseParser):
        '''
        注册新的解析器
        args:
            ext: 文件扩展名
            parser_cls: 继承自BaseParser的解析器类
        '''
        cls._parsers[ext.lower()] = parser_cls
        
    @classmethod
    def support_file_types(cls) -> List[str]:
        '''
        返回所有支持的文件扩展名
        '''
        return list(cls._parsers.keys())


#调用函数
def parse_file(file_path: str) -> ParseResult:
    '''
    调用函数
    根据文件路径解析文档
    '''
    parser = ParserFactory.get_parser(file_path)
    return parser.parse(file_path)