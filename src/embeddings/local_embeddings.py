from sentence_transformers import SentenceTransformer
from typing import List

from src.embeddings.base import Base

class LocalEmbeddings(Base):
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu"):
        """

        Args:
            model_name (str, optional): _description_. Defaults to "all-MiniLM-L6-v2".
            device (str, optional): _description_. Defaults to "cpu".
            model推荐
            all-MiniLM-L6-v2特点: 精度高，速度慢,适合大多数场景
            all-MiniLM-L12-v2特点: 精度更高，速度更慢，适合对精度要求较高的场景
            all-mpnet-base-v2特点: 精度很高，速度较慢，适合对精度要求很高的场景
            all-MiniLM-L6-v2-zh特点: 适用于中文场景
            all-distilroberta-v1特点: 速度较快，精度一般
            all-MiniLM-L6-a-snli-mpnet-base-v2特点: 适用于自然语言推理任务
            all-MiniLM-L6-a-snli-mpnet-base-v2-zh特点: 适用于中文自然语言推理任务

        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        
    #单个文本向量化
    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True, device=self.device)
        return embedding.tolist()
    
    #批量文本向量化
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True, device=self.device)
        return embeddings.tolist()
    
    #获取向量维度
    def get_dimension(self) -> int:
        # Base.get_dimension 是抽象方法（方法体只有 pass），不能靠 super() 取维度，
        # 必须向底层 SentenceTransformer 模型询问真实维度。
        # sentence-transformers >= 6.2 提供 get_embedding_dimension()。
        return self.model.get_embedding_dimension()
    
    #获取模型名称
    def get_model_name(self) -> str:
        return self.model_name