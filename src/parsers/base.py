#基础解析器
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import metadata
from typing import Dict,List


@dataclass
class ParseResult:
    '''
    解析结果统一格式
    '''
    file_name:str  #文件名称
    total_pages:int#文件总页数
    total_chunks:int #文本块数
    table_count:int #表格数
    chunks:List[Dict]#文本块列表
    raw_text:str  #原始文本
    tables:List[Dict] #表格列表
    metadata:Dict  #其他元数据
    pass

class BaseParser(ABC):
    '''
    基础解析器类
    '''
    @abstractmethod
    def parse(self,file_path:str) -> ParseResult:
        '''
        从文件路径解析文档
        :param file_path: 文件路径
        :return: 解析结果
        '''
        pass
    
    @abstractmethod
    def parse_from_bytes(self,file_bytes:bytes,file_name:str) -> ParseResult:
        '''
        从字节流解析文档
        :param file_bytes: 文件字节流
        :param file_name: 文件名
        :return: 解析结果
        '''
        pass
    
    @abstractmethod
    def support_file_types(self) -> list:
        '''
        返回支持的文件扩展名
        :return: 支持的扩展名列表,如['.pdf']
        '''
        pass
    
    
    

    