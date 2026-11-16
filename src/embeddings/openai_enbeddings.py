from typing import List
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI
from src.embeddings.base import Base


# 各模型在“未显式指定 dimension”时的默认向量维度。
# DashScope（阿里云百炼）text-embedding-v4 默认 1024，可选 2048/1536/1024/768/512/256/128/64；
# v3 默认 1024；v1/v2 为 1536。OpenAI 侧 ada-002 / 3-small 为 1536，3-large 为 3072。
_MODEL_DIMENSIONS = {
    "text-embedding-v4": 1024,
    "text-embedding-v3": 1024,
    "text-embedding-v2": 1536,
    "text-embedding-v1": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    # 硅基流动（SiliconFlow）：bge 系列免费；Qwen3-Embedding 支持 dimensions 参数，
    # 这里按平台允许的最大维度填写（4B 原生 2560，但平台可选上界为 2048）。
    "BAAI/bge-m3": 1024,
    "Pro/BAAI/bge-m3": 1024,
    "BAAI/bge-large-zh-v1.5": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "netease-youdao/bce-embedding-base_v1": 768,
    "Qwen/Qwen3-Embedding-0.6B": 1024,
    "Qwen/Qwen3-Embedding-4B": 2048,
    "Qwen/Qwen3-Embedding-8B": 4096,
}


class OpenAIEmbeddings(Base):
    def __init__(
        self,
        api_key: str, 
        model: str,
        base_url: str = "https://api.openai.com/v1",
        dimension:int |None=None
        ):
        if dimension is not None and dimension <= 0:
            raise ValueError(f"dimension 必须为正整数，当前为 {dimension}")
        # base_url 必须传给 OpenAI 客户端，否则请求会一直发往 api.openai.com，
        # 使用 DashScope 等兼容接口时会导致鉴权失败。
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dimension = dimension
        self.base_url = base_url
    # 单个嵌入（实现 Base 的抽象方法 embed_text）
    def embed_text(self, text: str) -> List[float]:
        #调用批量嵌入方法
        embeddings = self.embed_batch([text])
        return embeddings[0] if embeddings else []

    # 兼容旧接口名：embed 与 embed_text 等价
    embed = embed_text
    
    
    # 重试策略：wait_random_exponential(min=2, max=20) 随机指数退避，最多尝试 3 次。
    # 注意：multiplier=1 时前两次等待都会被 min=2 抬到 2 秒，并非固定的 2/4/8 秒。
    @retry(wait=wait_random_exponential(min=2, max=20), stop=stop_after_attempt(3))
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入
        :param texts: 批量文本
        :return: 批量嵌入结果
        """

        # 空列表直接返回，避免向真实接口发送空的 input
        if not texts:
            return []
        params = {"input": texts, "model": self.model}
        # dimension 为 None 表示使用模型默认维度；0/负数在构造时已被拒绝
        if self.dimension is not None:
            params["dimensions"] = self.dimension
        #发送请求
        response = self.client.embeddings.create(**params)
        embeddings = [item.embedding for item in response.data]
        return embeddings
    
    def get_dimension(self) -> int:
        """
        获取向量维度
        :return: 向量维度
        """
        if self.dimension is not None:
            return self.dimension
        # 未指定维度时按模型默认值兜底，未知模型按 DashScope 现行默认 1024 处理
        return _MODEL_DIMENSIONS.get(self.model, 1024)
        
    def get_model_name(self) -> str:
        """
        获取模型名称
        :return: 模型名称
        """
        return self.model