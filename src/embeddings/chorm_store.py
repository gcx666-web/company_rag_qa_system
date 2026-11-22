# -*- coding: utf-8 -*-
"""
chorm_store.py —— Chroma 向量数据库实现。

BaseVectorStore 的 Chroma 版本：把 Document + 向量写入本地持久化集合，
并提供检索、删除、清空和集合信息查询。

典型用法::

    store = ChromaVectorStore(persist_directory="./data/chroma_db")
    ids = store.add_documents(documents, embeddings)
    results = store.search(query_embedding, top_k=5)

写入前的归一化说明（Chroma 的数据约束）：

- id 必须是 str：上游 chunk 的整型序号会统一转成字符串，空 id 自动生成 uuid；
- metadata 只接受 str/int/float/bool 及其同类型列表：None 值和嵌套 dict/list
  会被丢弃（打 warning），避免整批写入因个别字段失败；
- documents / embeddings 必须等长且非空，写入前先校验，报错信息比 Chroma
  自带的 "Unequal lengths for fields" 更直观。
"""

import logging
import numbers
import uuid
from collections import Counter
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.errors import DuplicateIDError

from src.embeddings.vector_store import BaseVectorStore, Document, SearchResult

logger = logging.getLogger(__name__)

__all__ = ["ChromaVectorStore"]

# Chroma 支持的相似度空间：l2 欧氏距离、ip 内积、cosine 余弦距离
_SUPPORTED_SPACES = ("l2", "ip", "cosine")


class ChromaVectorStore(BaseVectorStore):
    """
    Chromadb向量数据库实现
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./data/chroma_db",  # 数据库保存路径
        distance: str = "cosine",
        client: Optional[Any] = None,
    ):
        """
        构造函数。

        :param collection_name:   集合名称。
        :param persist_directory: 本地持久化目录。
        :param distance:          相似度空间，仅支持 l2 / ip / cosine。
        :param client:            可选的已有 Chroma 客户端。正式运行链路由
                                  ``src.database.chroma_client.get_chroma_client``
                                  传入复用实例，避免每个请求都重建客户端；
                                  不传时保持旧行为（直接创建 PersistentClient），
                                  方便单元测试 mock。
        :raises ValueError:       distance 不在白名单内时抛出。
        """
        if distance not in _SUPPORTED_SPACES:
            raise ValueError(
                f"distance 只支持 {_SUPPORTED_SPACES}，当前传入 {distance!r}"
            )
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        # 注意：这里不能写成 distance,（尾部逗号会把字符串变成元组，
        # 导致 Chroma 报 "not a valid space: ('cosine',)"）
        self.distance = distance
        # 优先复用外部注入的客户端；没有注入时才自建（保持原调用链不变）
        self.client = client or chromadb.PersistentClient(
            path=self.persist_directory
        )
        # 获取或创建集合
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """
        内部工具：按当前配置获取或创建集合。

        元数据 hnsw:space 只在集合首次创建时生效；集合已存在时 Chroma 会忽略
        这里的 metadata 并沿用原配置。
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},  # 元数据：如何计算相似度
        )

    # ==================================================================
    # 一、写入
    # ==================================================================
    def add_documents(
        self, documents: List[Document], embeddings: List[List[float]]
    ) -> List[str]:
        """
        添加文档到向量存储。

        写入前会做三件事：id 归一化为 str、metadata 剔除 Chroma 不支持的值、
        校验 documents 与 embeddings 数量一致；自动生成的 id 会回写到
        Document.id，方便调用方后续按 id 删除。

        :param documents: 文档列表
        :param embeddings: 对应的向量列表（与 documents 等长）
        :return: 文档id列表
        :raises ValueError: 数量不一致、同批 id 重复时抛出
        :raises DuplicateIDError: 集合中已存在相同 id 时抛出
        """
        if not documents:
            # Chroma 对空列表会报 "Non-empty lists are required"，这里直接早退
            return []

        if len(documents) != len(embeddings):
            raise ValueError(
                "documents 与 embeddings 数量不一致："
                f"{len(documents)} 个文档 vs {len(embeddings)} 个向量"
            )

        ids = [self._normalize_id(document.id) for document in documents]
        duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)# 统计重复的id
        if duplicates:
            raise ValueError(
                f"同一批文档中出现重复 id，请先修复上游数据：{duplicates[:10]}"
            )

        metadatas = [
            self._normalize_metadata(document.metadata) for document in documents
        ]
        texts = [
            "" if document.content is None else str(document.content)
            for document in documents
        ]

        try:
            self.collection.add(
                ids=ids,
                documents=texts,# 文档内容
                embeddings=embeddings,
                metadatas=metadatas,# 文档元数据-如何计算相似度
            )
        except DuplicateIDError as exc:
            # 保留原异常类型，把英文报错换成能直接指导操作的提示
            raise DuplicateIDError(
                "向量库中已存在相同 id 的文档，重复导入前请先调用 "
                f"delete_document 删除旧记录；示例 id：{ids[:10]}"
            ) from exc

        # 自动生成的 id 回写，保证调用方拿到删除/更新所需的真实 id
        for document, document_id in zip(documents, ids):
            document.id = document_id
        return ids

    # ==================================================================
    # 二、检索
    # ==================================================================
    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[SearchResult]:
        """
        向量检索。

        :param query_embedding: 查询向量
        :param top_k: 返回的结果数量
        :return: 检索结果列表（按相似度从高到低）；集合为空时返回空列表
        :raises ValueError: 查询向量为空时抛出
        """
        if top_k <= 0:
            return []
        if not query_embedding:
            raise ValueError("query_embedding 不能为空")

        total = self.collection.count()
        if total == 0:
            # 空集合检索会报 "Number of requested results ... is greater than
            # number of elements in index 0"，这里直接返回空列表
            return []

        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        # query 的结果按“每个查询向量一组”返回，这里只查了一个向量，取第 0 组
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        results: List[SearchResult] = []
        for index, document_id in enumerate(ids):
            content = documents[index] if index < len(documents) else None
            metadata = metadatas[index] if index < len(metadatas) else None
            metadata = dict(metadata or {})
            distance = distances[index] if index < len(distances) else None
            extra_metadata = dict(metadata)
            if distance is not None:
                # 保留 Chroma 原始 distance，方便调用方自己定义阈值
                extra_metadata["distance"] = distance
            results.append(
                SearchResult(
                    document=Document(
                        id=str(document_id),
                        content="" if content is None else str(content),
                        metadata=metadata,
                    ),
                    score=self._distance_to_score(distance),
                    metadata=extra_metadata,
                )
            )
        return results

    # ==================================================================
    # 三、删除与清空
    # ==================================================================
    def delete_document(self, document_id: str) -> None:
        """
        删除文档。

        :param document_id: 文档id；为空或不存在时静默跳过
        """
        if document_id is None or not str(document_id).strip():
            logger.warning("delete_document 收到空的 document_id，已跳过")
            return
        self.collection.delete(ids=[str(document_id)])

    def delete_by_file_name(
        self, file_name: str, namespace: Optional[str] = None
    ) -> int:
        """
        按文件名删除分块，返回删除条数。

        Chroma 的 delete 只接受 id 列表，因此先用 metadata 过滤查出命中记录
        （include 只要 metadatas，不额外拉取文档与向量），再按 namespace
        在客户端过滤后批量删除；namespace 为 None 时删除所有命名空间下的同文件分块。

        :param file_name: 文件名（与写入时 metadata.file_name 一致）。
        :param namespace: 命名空间过滤；None 表示不限制。
        :return:          实际删除的分块条数。
        """
        if not file_name or not str(file_name).strip():
            logger.warning("delete_by_file_name 收到空文件名，已跳过")
            return 0

        result = self.collection.get(
            where={"file_name": str(file_name)}, include=["metadatas"]
        )
        ids = list(result.get("ids") or [])
        if namespace is not None:
            metadatas = result.get("metadatas") or []
            ids = [
                doc_id
                for index, doc_id in enumerate(ids)
                if index < len(metadatas)
                and (metadatas[index] or {}).get("namespace") == namespace
            ]
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def clear(self) -> bool:
        """
        清空向量存储：删除集合并按当前配置重建。

        :return: 清空是否成功  True: 成功  False: 失败
        """
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as exc:
            logger.error("删除集合 %s 失败：%s", self.collection_name, exc)
            return False
        try:
            self.collection = self._get_or_create_collection()
        except Exception as exc:
            logger.error("重建集合 %s 失败：%s", self.collection_name, exc)
            return False
        return True

    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取向量存储的集合信息。

        :return: 集合信息字典（名称、记录数、相似度空间、持久化目录）
        """
        return {
            "name": self.collection_name,
            "count": self.collection.count(),
            "distance": self.distance,
            "persist_directory": self.persist_directory,
        }

    def get_vector_dimension(self) -> Optional[int]:
        """读取集合中实际存储的向量维度；空集合或读取失败时返回 None。

        切换在线/离线向量模型后，旧集合的维度可能与当前模型不一致。知识库列表
        接口用它提示用户“需要重新索引”，避免上传/问答时才发现 Chroma 报错。
        """
        try:
            if self.collection.count() <= 0:
                return None
            data = self.collection.get(limit=1, include=["embeddings"])
        except Exception as exc:
            logger.warning("读取集合 %s 向量维度失败：%s", self.collection_name, exc)
            return None

        embeddings = data.get("embeddings")
        if embeddings is None:
            return None
        # 不同 Chroma 版本可能返回 numpy 数组或嵌套列表
        shape = getattr(embeddings, "shape", None)
        if shape is not None and len(shape) == 2:
            return int(shape[1])
        if len(embeddings) == 0:
            return None
        first = embeddings[0]
        try:
            return len(first)
        except TypeError:
            return None

    # ==================================================================
    # 四、文件级查询（知识库管理 / 秒传校验）
    # ==================================================================
    def has_file(self, file_name: str, namespace: Optional[str] = None) -> bool:
        """
        判断向量库中是否存在指定文件的分块。

        用途：上传登记表只记录“曾经成功导入过”，但用户可能已经手动删除了
        向量库里的分块；秒传前必须再向向量库确认一次，避免登记表与真实数据
        不一致时返回“重复上传成功”却查不到任何内容。

        :param file_name: 文件名（与 metadata.file_name 一致）。
        :param namespace: 命名空间过滤；None 表示不限制。
        """
        if not file_name or not str(file_name).strip():
            return False
        try:
            result = self.collection.get(
                where={"file_name": str(file_name)}, include=["metadatas"]
            )
        except Exception as exc:
            logger.warning("按文件名查询向量库失败：%s", exc)
            return False

        ids = list(result.get("ids") or [])
        if not ids:
            return False
        if namespace is None:
            return True
        metadatas = result.get("metadatas") or []
        return any(
            (metadata or {}).get("namespace") == namespace
            for metadata in metadatas
        )

    def list_files(self, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        列出向量库中的文件及其统计信息（按 namespace + file_name 聚合）。

        返回值示例::

            [{
                "file_name": "员工手册.pdf",
                "namespace": "hr",
                "chunks": 42,
                "tables": 3,
                "pages": [1, 2, 5],
                "total_chars": 18321,
            }]

        :param namespace: 只返回指定 namespace 的文件；None 表示全部。
        """
        try:
            result = self.collection.get(include=["metadatas"])
        except Exception as exc:
            logger.warning("列出向量库文件失败：%s", exc)
            return []

        metadatas = result.get("metadatas") or []
        files: Dict[tuple[str, str], Dict[str, Any]] = {}
        for metadata in metadatas:
            if not isinstance(metadata, dict):
                continue
            file_name = str(metadata.get("file_name") or "").strip()
            if not file_name:
                continue
            item_namespace = str(metadata.get("namespace") or "")
            if namespace is not None and item_namespace != namespace:
                continue
            key = (item_namespace, file_name)
            item = files.get(key)
            if item is None:
                item = {
                    "file_name": file_name,
                    "namespace": item_namespace,
                    "chunks": 0,
                    "tables": 0,
                    "pages": set(),
                    "total_chars": 0,
                }
                files[key] = item

            item["chunks"] += 1
            if metadata.get("chunk_type") == "table":
                item["tables"] += 1
            page = metadata.get("page")
            if page is not None:
                try:
                    item["pages"].add(int(page))
                except (TypeError, ValueError):
                    pass
            char_count = metadata.get("char_count")
            if isinstance(char_count, (int, float)):
                item["total_chars"] += int(char_count)

        # set 不能直接 JSON 序列化，转成排序后的列表
        result_list = []
        for item in files.values():
            item["pages"] = sorted(item["pages"])
            result_list.append(item)
        return sorted(
            result_list, key=lambda value: (value["namespace"], value["file_name"])
        )

    # ==================================================================
    # 五、内部工具
    # ==================================================================
    @staticmethod
    def _normalize_id(raw_id: Any) -> str:
        """
        内部工具：把任意 id 归一化成 Chroma 要求的非空字符串。

        - None / 空串 / 纯空白 → 自动生成 uuid4；
        - 其余（含上游 chunk 的整型序号）→ 转成 str。
        """
        if raw_id is None:
            return str(uuid.uuid4())
        text = str(raw_id).strip()
        return text or str(uuid.uuid4())

    @staticmethod
    def _normalize_metadata(
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        内部工具：只保留 Chroma 能接受的 metadata 字段。

        Chroma 仅支持 str / int / float / bool 标量值，遇到 None 或嵌套
        dict/list 会让整批写入失败，因此这里统一丢弃并记录 warning；
        归一化后为空时返回 None（Chroma 允许单条记录的 metadata 为 None）。
        """
        if not metadata:
            return None
        if not isinstance(metadata, dict):
            logger.warning(
                "metadata 不是 dict（%s），本条记录不写入 metadata",
                type(metadata).__name__,
            )
            return None

        clean: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                logger.warning("metadata 字段 %s 的值为 None，已丢弃", key)
                continue
            key = str(key)
            if isinstance(value, str) or isinstance(value, bool):
                clean[key] = value
            elif isinstance(value, numbers.Integral):
                clean[key] = int(value)
            elif isinstance(value, numbers.Real):
                clean[key] = float(value)
            else:
                logger.warning(
                    "metadata 字段 %s 的值类型为 %s，Chroma 不支持，已丢弃",
                    key,
                    type(value).__name__,
                )
        return clean or None

    def _distance_to_score(self, distance: Optional[float]) -> float:
        """
        内部工具：把 Chroma 的 distance 换算成“越大越相似”的 score。

        - cosine / ip 空间：distance = 1 - 相似度，score 直接取 1 - distance；
        - l2 空间：Chroma 返回平方欧氏距离，用 1 / (1 + distance) 映射到 (0, 1]。
        """
        if distance is None:
            return 0.0
        distance = float(distance)
        if self.distance == "l2":
            return 1.0 / (1.0 + distance)
        return 1.0 - distance

