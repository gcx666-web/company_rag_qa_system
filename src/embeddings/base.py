from abc import ABC, abstractmethod
class Base(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """嵌入单个文本字符串为向量"""
        pass

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量嵌入文本字符串为向量"""
        pass
    #获取向量维度
    @abstractmethod
    def get_dimension(self) -> int:
        """获取向量维度"""
        pass
    #获取模型名称
    @abstractmethod
    def get_model_name(self) -> str:
        """获取模型名称"""
        pass