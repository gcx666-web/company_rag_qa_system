# -*- coding: utf-8 -*-
"""
local_embeddings.py —— 本地（离线）向量模型。

本文件在保留原有 ``LocalEmbeddings`` 用法的基础上，补齐了三项能力：

1. 离线保障：加载模型前设置 HuggingFace 离线环境变量，并默认
   ``local_files_only=True``，确保运行期不会偷偷访问外网；
2. 懒加载：``lazy=True`` 时构造对象不加载模型，第一次真正向量化时才加载，
   避免 FastAPI 启动被几 GB 的模型加载拖住；
3. 线程安全：模型加载与推理分别加锁，多线程并发上传时不会重复初始化或
   出现 PyTorch 推理竞争。

注意：为了兼容现有单元测试，默认构造（``lazy=False``、``offline=False``）
仍然保持“立即创建 SentenceTransformer、按原始参数调用 encode”的行为。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from src.embeddings.base import Base

logger = logging.getLogger(__name__)

__all__ = ["LocalEmbeddings"]


class LocalEmbeddings(Base):
    """基于 SentenceTransformer 的本地向量模型。

    :param model_name:        HuggingFace 模型名或本地模型目录。
    :param device:            推理设备，如 cpu / cuda / cuda:0 / mps。
    :param lazy:              是否延迟到第一次向量化时才加载模型（默认 False）。
    :param offline:           是否强制离线；True 时设置 HF 离线变量并只读本地缓存。
    :param local_files_only:  是否只使用本地文件；None 表示跟随 offline。
    :param cache_folder:      HuggingFace 缓存目录；None 使用默认缓存。
    :param batch_size:        批量推理时每批的文本条数。
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        *,
        lazy: bool = False,
        offline: bool = False,
        local_files_only: Optional[bool] = None,
        cache_folder: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.offline = bool(offline)
        self.local_files_only = (
            bool(local_files_only) if local_files_only is not None else self.offline
        )
        self.cache_folder = cache_folder
        self.batch_size = max(1, int(batch_size))

        # 模型对象在加载完成前为 None；保持公开的 ``model`` 属性以兼容旧代码
        self.model: Optional[SentenceTransformer] = None
        self._load_lock = threading.RLock()
        self._infer_lock = threading.Lock()
        self._loaded = False

        if not lazy:
            self._ensure_model()

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------
    def _apply_offline_env(self) -> None:
        """在加载模型前设置离线环境变量。

        这些变量会被 huggingface_hub / transformers 在下载或解析模型时读取，
        设置得越早越可靠；即使模型已经缓存，也能避免额外的版本检查请求。
        """
        if not self.offline:
            return
        for name in (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
        ):
            os.environ[name] = "1"
        # 关闭遥测与软链接警告，减少离线环境下的无意义日志
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    def _ensure_model(self) -> SentenceTransformer:
        """线程安全地加载模型；已加载时直接返回。"""
        if self._loaded and self.model is not None:
            return self.model
        with self._load_lock:
            if self._loaded and self.model is not None:
                return self.model

            self._apply_offline_env()
            model_path = self._resolve_model_path()

            # 基础参数与旧实现保持一致（单元测试依赖这两个关键字参数）
            kwargs: dict = {"device": self.device}
            # 只有显式离线 / 指定只读本地文件时才追加参数，避免改变默认调用链
            if self.local_files_only:
                kwargs["local_files_only"] = True
            if self.cache_folder:
                kwargs["cache_folder"] = self.cache_folder

            logger.info(
                "加载本地向量模型：%s（device=%s, offline=%s, local_only=%s）",
                model_path,
                self.device,
                self.offline,
                self.local_files_only,
            )
            try:
                self.model = SentenceTransformer(model_path, **kwargs)
            except Exception as exc:
                # 给出可操作的提示，而不是把 HuggingFace 的长堆栈直接抛给用户
                if self.offline:
                    raise RuntimeError(
                        f"离线模式加载向量模型失败：{model_path}。"
                        "请确认模型已下载到本地缓存，或用 OFFLINE_EMBEDDING_MODEL / "
                        "OFFLINE_EMBEDDING_DIR 指向已有的本地模型目录。"
                        f"原始错误：{type(exc).__name__}: {exc}"
                    ) from exc
                raise
            self._loaded = True
            logger.info("本地向量模型加载完成：%s", model_path)
            return self.model

    def _resolve_model_path(self) -> str:
        """解析模型路径：优先直接使用存在的本地目录，否则按模型名交给 HF 缓存。"""
        candidate = str(self.model_name or "").strip()
        if candidate and os.path.isdir(candidate):
            return candidate
        return candidate

    def _encode(self, texts):
        """内部编码：按 batch_size 分段，单段时保持一次 encode 调用。"""
        model = self._ensure_model()
        with self._infer_lock:
            if len(texts) <= self.batch_size:
                return model.encode(
                    texts, convert_to_numpy=True, device=self.device
                )
            # 超大批量分段推理，降低峰值内存
            chunks = []
            for start in range(0, len(texts), self.batch_size):
                chunks.append(
                    model.encode(
                        texts[start : start + self.batch_size],
                        convert_to_numpy=True,
                        device=self.device,
                    )
                )
            # 拼接多个 numpy 数组；没有 numpy 时退化为列表拼接
            try:
                import numpy as np

                return np.concatenate(chunks, axis=0)
            except Exception:
                result = []
                for chunk in chunks:
                    result.extend(
                        chunk.tolist() if hasattr(chunk, "tolist") else list(chunk)
                    )
                return result

    @staticmethod
    def _to_list(encoded) -> list:
        """把 numpy 数组 / 张量 / 列表统一转成 Python list。"""
        if hasattr(encoded, "tolist"):
            return encoded.tolist()
        return list(encoded)

    # ------------------------------------------------------------------
    # Base 接口实现
    # ------------------------------------------------------------------
    def embed_text(self, text: str) -> List[float]:
        """单条文本向量化（保持旧实现的 encode 调用参数）。"""
        embedding = self._encode(text)
        values = self._to_list(embedding)
        # 某些模型对单条输入会返回 [[...]]，这里统一压平为一维
        if values and isinstance(values[0], list):
            return [float(value) for value in values[0]]
        return [float(value) for value in values]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量文本向量化。"""
        if texts is None:
            return []
        embeddings = self._encode([str(text) for text in texts])
        values = self._to_list(embeddings)
        if not values:
            return []
        # 单条输入被 encode 成 [dim] 时补一层，保证返回值始终是二维
        if not isinstance(values[0], list):
            values = [values]
        return [[float(value) for value in row] for row in values]

    def get_dimension(self) -> int:
        """获取向量维度（需要先加载模型）。"""
        model = self._ensure_model()
        return int(model.get_embedding_dimension())

    def get_model_name(self) -> str:
        """获取模型名称（本地权重目录时返回目录名）。"""
        return self.model_name

    # ------------------------------------------------------------------
    # 诊断 / 预热
    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """模型是否已经加载到内存。"""
        return self._loaded

    def warmup(self) -> int:
        """预热模型并返回向量维度，用于 /api/system/embedding/warmup。"""
        vector = self.embed_text("预热")
        return len(vector) if vector else self.get_dimension()

    def info(self) -> dict:
        """返回模型状态信息（不触发懒加载）。"""
        return {
            "model": self.model_name,
            "device": self.device,
            "offline": self.offline,
            "local_files_only": self.local_files_only,
            "loaded": self._loaded,
            "dimension": self.get_dimension() if self._loaded else None,
        }
