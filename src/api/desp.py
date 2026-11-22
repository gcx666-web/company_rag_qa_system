#依赖注入：向量模型 / 文档存储 / 检索服务 / LLM 实例
import os

from dotenv import load_dotenv

from src.api.config import settings
from src.database.document_store import DocumentStore
from src.embeddings.chorm_store import ChromaVectorStore
from src.embeddings.openai_enbeddings import OpenAIEmbeddings
from src.llm.llm import LLM
from src.retrieval.search_service import SearchService

load_dotenv()


def get_embedding_model() -> OpenAIEmbeddings:
    '''
    获取向量模型实例

    密钥优先级：EMBEDDING_API_KEY 回退 SILICONFLOW_API_KEY，
    模型与地址统一由 EMBEDDING_MODEL / EMBEDDING_BASE_URL 指定。
    :raises ValueError: 两个密钥都没有配置时抛出（避免下游只报“缺少凭据”）
    '''
    api_key = (
        os.getenv("EMBEDDING_API_KEY")
        or os.getenv("SILICONFLOW_API_KEY")
        or ""
    )
    if not api_key:
        raise ValueError(
            "未配置向量模型密钥：请在 .env 中设置 EMBEDDING_API_KEY（或 SILICONFLOW_API_KEY）"
        )
    return OpenAIEmbeddings(
        api_key=api_key,
        model=os.getenv("EMBEDDING_MODEL") or "",
        base_url=os.getenv("EMBEDDING_BASE_URL") or "",
    )


def get_document_store(collection_name: str) -> DocumentStore:
    '''
    获取文档存储实例

    :param collection_name: Chroma 集合名称，同时作为写入/检索使用的 namespace
    '''
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )
    return DocumentStore(
        embedding_model=get_embedding_model(),
        vector_store=vector_store,
        collection_name=collection_name,
        persist_directory=settings.CHROMA_PERSIST_DIR,
    )


def get_search_service(collection_name: str) -> SearchService:
    '''
    获取检索服务实例
    '''
    return SearchService(document_store=get_document_store(collection_name))


def get_llm(model: str) -> LLM:
    '''
    获取llm实例

    :param model: 模型来源: zhipu, qwen, siliconflow
    '''
    if model not in settings.LLM_PROVIDER:
        raise ValueError(f"Invalid model: {model}")
    model_config = settings.LLM_PROVIDER[model]
    return LLM(
        api_key=model_config["API_KEY"] or "",
        base_url=model_config["BASE_URL"] or "",
        model=model_config["MODEL"] or "",
    )