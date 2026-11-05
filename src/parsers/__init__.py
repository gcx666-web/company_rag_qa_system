from pathlib import Path
from src.parsers.base import BaseParser, ParseResult
from src.parsers.pdf_parser import PdfParser
from src.parsers.word_parser import WordParser
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.htmlparser import HtmlParser

from typing import List


class ParserFactory:
    '''
    解析器工厂类    
    '''
    _parsers = {
        '.pdf': PdfParser,
        '.docx': WordParser,
        '.md': MarkdownParser,
        '.html':HtmlParser
    }
    
    @classmethod
    def get_parser(cls, file_path: str) -> BaseParser:
        '''
        根据文件路径获取对应的解析器
        '''
        ext = Path(file_path).suffix.lower()
        for parser_cls in cls._parsers.values():
            if ext in parser_cls().support_file_types():
                return parser_cls()
        raise ValueError(f"不支持的文件类型: {ext}")
    
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
        return list(cls._parsers.keys())  # 去重


#调用函数
def parse_file(file_path: str) -> ParseResult:
    '''
    调用函数
    根据文件路径解析文档
    '''
    parser = ParserFactory.get_parser(file_path)
    return parser.parse(file_path)