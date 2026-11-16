# -*- coding: utf-8 -*-
"""
document_store.py —— 文档存储服务。

把“解析结果 / 纯文本”到“向量库”的链路收口在一个服务里：

    解析器 ParseResult ──► add_parse_result ──► 分块转 Document
                                                    │
                                    embedding_model.embed_batch
                                                    ▼
                              ChromaVectorStore.add_documents

    查询文本 ──► search ──► embed_text ──► 向量检索 ──► 结果字典

设计约定：

- 依赖抽象：embedding_model 只要求实现 Base（embed_text / embed_batch /
  get_dimension / get_model_name），向量库默认使用 ChromaVectorStore；
- 文档 id 是确定性的：``{namespace}_{file_name}_{chunk_id}``，便于按文件名 +
  分块序号溯源。注意 Chroma 对已存在的 id 是“静默忽略”（既不报错也不覆盖），
  因此本服务在写入前显式检查 id，命中已存在记录时抛 DuplicateIDError；重复
  导入请先调用 delete_document 删除旧分块，避免出现“看起来写入成功、实际还是
  旧内容”的假象；
- namespace 用于区分不同文档集：写入时落到 metadata.namespace，检索时按它过滤；
- 本服务只做字段映射与流程编排，metadata 清洗（丢弃 None / 嵌套结构）仍由
  ChromaVectorStore 统一处理，这里不重复实现。
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from chromadb.errors import DuplicateIDError

from src.embeddings.base import Base
from src.embeddings.chorm_store import ChromaVectorStore
from src.embeddings.vector_store import Document, SearchResult
from src.parsers.base import ParseResult

logger = logging.getLogger(__name__)

__all__ = ["DocumentStore"]


class DocumentStore:
    """文档存储服务：整合解析、向量化、存储。"""

    # metadata 中命名空间字段名；写入与过滤共用同一常量，避免拼写漂移
    NAMESPACE_KEY = "namespace"

    def __init__(
        self,
        embedding_model: Base,
        vector_store: Optional[ChromaVectorStore] = None,
        collection_name: str = "rag_documents",
        persist_directory: str = "./data/chroma_db",
    ):
        """
        初始化文档存储。

        :param embedding_model:   嵌入模型（必需），需实现 Base 的四个抽象方法。
        :param vector_store:      向量存储；不传时按 collection_name /
                                  persist_directory 自建 ChromaVectorStore。
        :param collection_name:   自建向量库时的集合名称。
        :param persist_directory: 自建向量库时的本地持久化目录。
        :raises ValueError:       embedding_model 为空时抛出。
        """
        if embedding_model is None:
            raise ValueError("embedding_model 不能为空")
        self.embedding_model = embedding_model
        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
        )

    # ==================================================================
    # 一、写入：解析结果
    # ==================================================================
    def add_parse_result(
        self,
        parse_result: ParseResult,
        namespace: str = "default",
    ) -> List[str]:
        """
        添加解析结果到向量库。

        把 ParseResult.chunks（约定字段 type / content / page / chunk_id /
        metadata）逐个转成 Document，统一批量向量化后一次写入向量库。

        - 文档 id 为 ``{namespace}_{file_name}_{chunk_id}``；写入前会检查这些 id
          是否已存在，命中则抛 DuplicateIDError（Chroma 原生 add 对已存在 id 是
          静默忽略，会让人误以为更新成功），重复导入需先删除旧分块；
        - 空白内容、非 dict 的分块会被跳过（记录 warning），避免写入无意义向量；
        - 分块自带的 metadata（type_index / table_index / cols / row_count /
          part / total_parts 等）会一并保留，便于检索后溯源。

        :param parse_result: 文档解析结果。
        :param namespace:    命名空间（用于区分不同文档集）。
        :return:             文档 id 列表；没有可写入内容时返回空列表。
        :raises DuplicateIDError: 写入前检测到 id 已存在时抛出（由本服务显式检查）。
        """
        if parse_result is None:
            logger.warning("add_parse_result 收到空的 parse_result，已跳过")
            return []

        chunks = getattr(parse_result, "chunks", None) or []
        file_name = str(getattr(parse_result, "file_name", "") or "")
        if not chunks:
            logger.warning(
                "parse_result.chunks 为空（file_name=%r），没有可写入的内容", file_name
            )
            return []

        documents: List[Document] = []
        texts_to_embed: List[str] = []
        for position, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                logger.warning("第 %d 个分块不是 dict，已跳过：%r", position, chunk)
                continue
            content = chunk.get("content")
            content = "" if content is None else str(content)
            if not content.strip():
                logger.warning("第 %d 个分块内容为空，已跳过", position)
                continue

            chunk_id = self._resolve_chunk_id(chunk, position)
            chunk_type = str(chunk.get("type") or "text")
            metadata = self._build_chunk_metadata(
                file_name=file_name,
                chunk_id=chunk_id,
                chunk_type=chunk_type,
                content=content,
                chunk=chunk,
                namespace=namespace,
            )
            documents.append(
                Document(
                    id=f"{namespace}_{file_name}_{chunk_id}",
                    content=content,
                    metadata=metadata,
                )
            )
            texts_to_embed.append(content)

        if not documents:
            logger.warning(
                "解析结果中所有分块都无效（file_name=%r），没有写入任何记录", file_name
            )
            return []

        # Chroma 对已存在 id 的写入是静默忽略，写入前先拦截，避免“看起来成功但没更新”
        self._ensure_ids_absent([document.id for document in documents])

        # 一次性批量生成向量，再一次批量写入，避免逐块往返
        embeddings = self.embedding_model.embed_batch(texts_to_embed)
        return self.vector_store.add_documents(documents, embeddings)

    @staticmethod
    def _resolve_chunk_id(chunk: Dict[str, Any], position: int) -> Any:
        """
        内部工具：取分块的全局序号；上游没给 chunk_id 时退回列表下标。

        返回值只用于拼文档 id 和写 metadata（最终都会被转成字符串/标量）。
        """
        chunk_id = chunk.get("chunk_id")
        return position if chunk_id is None else chunk_id

    @classmethod
    def _build_chunk_metadata(
        cls,
        *,
        file_name: str,
        chunk_id: Any,
        chunk_type: str,
        content: str,
        chunk: Dict[str, Any],
        namespace: str,
    ) -> Dict[str, Any]:
        """
        内部工具：组装单个分块的 metadata。

        分块器自带的溯源字段先落地，再用本次写入的权威字段覆盖
        （file_name 以 ParseResult 为准，char_count 以真实内容长度为准）。
        Chroma 不支持的 None / 嵌套结构由 ChromaVectorStore 统一剔除。
        """
        metadata: Dict[str, Any] = {}
        raw_metadata = chunk.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata.update(raw_metadata)
        elif raw_metadata is not None:
            logger.warning(
                "分块 %r 的 metadata 不是 dict，已忽略：%r", chunk_id, raw_metadata
            )

        metadata.update(
            {
                "file_name": file_name,
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,
                "char_count": len(content),
                cls.NAMESPACE_KEY: namespace,
            }
        )
        # 页码只有表格块（或带页信息的解析器）才有，缺失时不写 None，保持 metadata 干净
        page = chunk.get("page")
        if page is not None:
            metadata["page"] = page
        return metadata

    def _ensure_ids_absent(self, ids: List[str]) -> None:
        """
        内部工具：写入前拦截已存在的文档 id。

        Chroma 的 collection.add 对已存在的 id 是“静默忽略”（既不报错也不覆盖，
        旧记录原样保留），直接重复导入会让人误以为更新成功。这里显式查一次已存在
        的 id 并抛 DuplicateIDError，把问题暴露在导入阶段（而不是事后翻库才发现
        内容没变）。

        说明：依赖 ChromaVectorStore 暴露的 collection 属性；若换用其他向量库
        实现（没有该属性）则跳过检查，由该实现自行决定重复 id 的语义。

        :param ids: 本次准备写入的文档 id 列表。
        :raises DuplicateIDError: 任一 id 已存在时抛出。
        """
        if not ids:
            return
        collection = getattr(self.vector_store, "collection", None)
        if collection is None:
            return
        # include=[] 只返回命中的 id，不额外拉取文档与向量
        existing = collection.get(ids=list(ids), include=[])
        duplicated = sorted(str(item) for item in (existing.get("ids") or []))
        if duplicated:
            raise DuplicateIDError(
                "向量库中已存在相同 id 的文档，重复导入前请先调用 "
                f"delete_document 删除旧记录；示例 id：{duplicated[:10]}"
            )

    # ==================================================================
    # 二、写入：纯文本
    # ==================================================================
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        namespace: str = "default",
    ) -> List[str]:
        """
        直接添加文本到向量库（不经过解析器/分块器）。

        - 每条文本生成独立的随机 id：``{namespace}_{uuid4.hex}``；
        - metadatas 短于 texts 时，缺失项按空字典处理；
        - metadata 会复制一份再写 namespace，不会修改调用方传入的字典；
        - 空白文本会被跳过（记录 warning）。

        :param texts:     文本列表。
        :param metadatas: 元数据列表（可为空列表）。
        :param namespace: 命名空间。
        :return:          文档 id 列表；没有可写入内容时返回空列表。
        """
        if not texts:
            logger.warning("add_documents 收到的 texts 为空，已跳过")
            return []

        metadatas = metadatas or []
        documents: List[Document] = []
        texts_to_embed: List[str] = []
        for i, text in enumerate(texts):
            content = "" if text is None else str(text)
            if not content.strip():
                logger.warning("add_documents 第 %d 条文本内容为空，已跳过", i)
                continue
            # 复制一份再写 namespace，避免污染调用方持有的字典
            metadata = metadatas[i] if i < len(metadatas) else {}
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            metadata[self.NAMESPACE_KEY] = namespace
            documents.append(
                Document(
                    # 用完整 uuid4.hex，避免短随机串在大量写入时发生 id 碰撞
                    id=f"{namespace}_{uuid.uuid4().hex}",
                    content=content,
                    metadata=metadata,
                )
            )
            texts_to_embed.append(content)

        if not documents:
            logger.warning("add_documents 没有可写入的文本（全部为空），已跳过")
            return []

        embeddings = self.embedding_model.embed_batch(texts_to_embed)
        return self.vector_store.add_documents(documents, embeddings)

    # ==================================================================
    # 三、检索
    # ==================================================================
    def search(
        self,
        query: str,
        top_k: int = 5,
        namespace: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义搜索。

        底层 ChromaVectorStore.search 不支持元数据过滤，因此 namespace 过滤在
        本服务内完成：先按 top_k 抓取，命中不足时逐步扩大抓取量（最多到集合
        总条数），再截取前 top_k 条，保证同 namespace 的记录较少时也能找齐。

        :param query:     查询文本；空白查询直接返回空列表。
        :param top_k:     返回结果数；非正数直接返回空列表。
        :param namespace: 命名空间过滤；None 表示不限制。
        :return:          结果字典列表（按相似度降序）：
                          {"id", "content", "score", "distance", "metadata"}。
        """
        if top_k <= 0:
            return []
        query = "" if query is None else str(query)
        if not query.strip():
            logger.warning("search 收到空 query，返回空结果")
            return []

        query_embedding = self.embedding_model.embed_text(query)
        if not query_embedding:
            logger.warning("embed_text 返回空向量（query=%r），返回空结果", query)
            return []

        if namespace is None:
            results = self.vector_store.search(
                query_embedding=query_embedding, top_k=top_k
            )
        else:
            results = self._search_in_namespace(query_embedding, top_k, namespace)

        return [
            {
                "id": r.document.id,
                "content": r.document.content,
                "score": r.score,
                # Chroma 的原始 distance 保留下来，便于调用方自定义相似度阈值
                "distance": (r.metadata or {}).get("distance"),
                "metadata": r.document.metadata,
            }
            for r in results
        ]

    def _search_in_namespace(
        self,
        query_embedding: List[float],
        top_k: int,
        namespace: str,
    ) -> List[SearchResult]:
        """
        内部工具：带 namespace 过滤的检索（客户端过滤 + 逐步扩大抓取量）。

        :param query_embedding: 查询向量。
        :param top_k:           需要返回的同 namespace 结果数。
        :param namespace:       目标命名空间。
        :return:                过滤后的检索结果（最多 top_k 条）。
        """
        total = int(self.vector_store.get_collection_info().get("count", 0) or 0)
        if total <= 0:
            return []

        fetch_k = min(total, top_k)
        while True:
            results = self.vector_store.search(
                query_embedding=query_embedding, top_k=fetch_k
            )
            matched = [
                r
                for r in results
                if (r.document.metadata or {}).get(self.NAMESPACE_KEY) == namespace
            ]
            # 抓够了，或者已经抓完整个集合（再扩大也不会有新记录）
            if len(matched) >= top_k or fetch_k >= total:
                return matched[:top_k]
            # fetch_k < total 且每轮至少 +1，不会死循环
            fetch_k = min(total, fetch_k * 4)

    # ==================================================================
    # 四、统计与维护
    # ==================================================================
    def get_stats(self) -> Dict[str, Any]:
        """
        获取存储统计信息。

        :return: ChromaVectorStore.get_collection_info() 的全部字段
                 （name / count / distance / persist_directory），
                 追加 embedding_model 与 embedding_dimension。
        """
        stats = dict(self.vector_store.get_collection_info() or {})
        stats["embedding_model"] = self.embedding_model.get_model_name()
        stats["embedding_dimension"] = self.embedding_model.get_dimension()
        return stats

    def delete_document(self, document_id: str) -> None:
        """
        删除单条文档（重复导入同一文件前，先删旧分块）。

        :param document_id: 文档 id；为空或不存在时由底层静默跳过。
        """
        self.vector_store.delete_document(document_id)

    def clear(self) -> bool:
        """
        清空向量存储（删除集合并按当前配置重建）。

        :return: 是否成功。
        """
        return self.vector_store.clear()

