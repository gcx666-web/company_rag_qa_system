from abc import ABC, abstractmethod
from typing import Dict,List, Any
from dataclasses import dataclass


@dataclass
class Document(ABC):
    """
    向量文档对象
    """
    id: str # 文档id
    content: str # 文档内容
    metadata: Dict[str, Any] # 文档元数据
    embedding: List[float] | None = None # 文档向量[可选]
    



@dataclass
class SearchResult:
    """
    向量检索结果
    """
    document: Document  # 文档对象
    score: float  # 相似度得分
    metadata: Dict[str, Any] # 额外元数据
    


class BaseVectorStore(ABC):
    """
    向量存储抽象类
    向量数据库接口
    """
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]) -> List[str]:
        """
        添加文档到向量存储
        :param documents: 文档列表
        :param embeddings: 对应的向量列表
        """
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """
        向量检索
        :param query_embedding: 查询向量
        :param top_k: 返回的结果数量
        :return: 检索结果列表
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        """
        删除文档
        :param document_id: 文档id
        """
        pass

    @abstractmethod
    def clear(self) -> bool:
        """
        清空向量存储
        :return: 清空是否成功  True: 成功  False: 失败  
        
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取向量存储的集合信息
        :return: 集合信息字典
        """
        pass