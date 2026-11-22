# -*- coding: utf-8 -*-
"""api/utils.py —— 上传/集合相关的通用工具函数。"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Union

from fastapi import HTTPException

__all__ = [
    "validate_collection_name",
    "sha256_bytes",
    "safe_filename",
    "atomic_write_bytes",
    "embedding_dimension_hint",
]

# Chroma 维度不匹配报错示例：
# Collection expecting embedding with dimension of 1024, got 512
_DIMENSION_ERROR = re.compile(
    r"expecting embedding with dimension of\s+(\d+),\s*got\s+(\d+)", re.IGNORECASE
)

# Chroma 集合名约束：3-512 个字符，只允许 [a-zA-Z0-9._-]，且首尾必须是字母或数字
_COLLECTION_NAME_CHARS = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def validate_collection_name(name: Union[str, None]) -> str:
    """校验 Chroma 集合名，非法时直接返回 400。

    下游 Chroma 对不合法的集合名会抛 InvalidArgumentError；如果不在入口拦下，
    就会被包装成笼统的“文件处理失败”，用户很难定位。这里把校验规则集中到一个
    函数，上传、创建集合、删除集合等接口共用。

    :param name: 原始集合名。
    :return: 去掉首尾空白后的集合名。
    :raises HTTPException: 400，附带具体原因。
    """
    name = str(name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="集合名称不能为空")
    if not 3 <= len(name) <= 512:
        raise HTTPException(
            status_code=400, detail="集合名称长度必须在 3-512 个字符之间"
        )
    if set(name) - _COLLECTION_NAME_CHARS:
        raise HTTPException(
            status_code=400,
            detail="集合名称只能包含字母、数字、下划线、点和短横线",
        )
    if not (name[0].isalnum() and name[-1].isalnum()):
        raise HTTPException(
            status_code=400, detail="集合名称必须以字母或数字开头和结尾"
        )
    return name


def sha256_bytes(data: bytes) -> str:
    """计算字节内容的 SHA-256 十六进制摘要（文件去重指纹）。"""
    return hashlib.sha256(data).hexdigest()


def safe_filename(raw_name: Union[str, None]) -> str:
    """从上传文件名中取出安全的纯文件名，避免路径穿越。

    - 同时处理 ``/`` 与 ``\\``，兼容浏览器在不同系统上传的文件名；
    - 去掉空字符、首尾空白；
    - 拒绝 ``.`` / ``..`` 等特殊名。

    :raises HTTPException: 400，文件名为空或非法。
    """
    name = str(raw_name or "").replace("\\", "/").split("/")[-1]
    name = name.replace("\x00", "").strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="文件名不能为空或非法")
    # Windows 文件名不允许以下字符；替换成下划线，尽量保留可读性
    name = re.sub(r'[<>:"|?*]', "_", name)
    return name


def embedding_dimension_hint(exc: Exception, collection_name: str) -> str | None:
    """把 Chroma 的向量维度报错翻译成可操作的中文提示。

    典型触发场景：知识库是在在线模型（如 1024 维）下建立的，之后切换到离线
    ``BAAI/bge-small-zh``（512 维），旧集合无法接收/检索新向量。Chroma 的原始
    报错是英文且埋在 500 里，用户看不出该怎么处理。

    :return: 命中维度错误时返回中文提示；不是维度错误则返回 None。
    """
    text = str(exc or "")
    match = _DIMENSION_ERROR.search(text)
    if not match:
        return None
    expected, got = match.group(1), match.group(2)
    return (
        f"知识库「{collection_name}」的向量维度为 {expected}，"
        f"当前向量模型输出 {got} 维，两者不一致。"
        "通常是切换了在线/离线向量模型但旧文档未重新索引导致的："
        "请新建一个知识库重新上传文档，或切回建立该知识库时使用的向量模型。"
    )


def atomic_write_bytes(target: Union[str, Path], data: bytes) -> Path:
    """原子写文件：先写同目录临时文件，再 ``os.replace`` 覆盖目标。

    好处：即使写入过程中进程崩溃，也不会留下半截文件；并发上传同名文件时，
    最终落盘的永远是一份完整内容。

    :param target: 目标路径。
    :param data:   文件字节内容。
    :return:       最终写入的绝对路径。
    """
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    finally:
        # os.replace 成功后临时文件已不存在；失败时清理残留
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
    return path.resolve()
