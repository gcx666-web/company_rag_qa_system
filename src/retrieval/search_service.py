from math import e
from typing import Any, Dict, List
from dataclasses import dataclass


from src.database.document_store import DocumentStore
import time
@dataclass
class RetrievalConfig:
    """"
    检索配置
    """
    top_k: int = 5 # 返回数量
    min_score: float = 0.0 # 最低分数阈值
    namespace: str | None= None # 命名空间
    metadata_filter:dict | None = None # 元数据过滤条件
    
    
@dataclass
class SearchResponse:
    """
    检索结果
    """
    #查询文本
    #总结果数量
    #检验结果
    #检验时间
    query: str
    total: int
    results: List[Dict[str, Any]]
    time: float = 0.0
    

class SearchService:
    """
    检索服务
    参数：
     document_store: 文档存储
     命名空间
     元数据过滤条件
     top_k: 返回数量
     返回：SearchResponse(
            query=query,
            total=len(results),
            results=results,
            time=total_time,
        )
     
    """
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store
        
        
    def search(self, query: str, config: RetrievalConfig |None=None) -> SearchResponse:
        start = time.time()
        if config is None:
            config = RetrievalConfig()
        results = self.document_store.search(query, top_k=config.top_k, namespace=config.namespace)
        results = [result for result in results if result["score"] >= config.min_score]
        #topk
        results = results[:config.top_k]
        end = time.time()
        total_time=(end-start)*1000
        return SearchResponse(
            query=query,
            total=len(results),
            results=results,
            time=total_time,
        )
        