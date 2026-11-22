# -*- coding: utf-8 -*-
"""
config.py —— 应用配置中心。

所有可调参数都集中在这里，并通过环境变量（.env）覆盖，避免散落在各个模块里
“魔法数字”。本文件新增的配置主要围绕四类优化：

1. 向量模型缓存：内存 LRU + SQLite 磁盘缓存，重复文本不再请求向量接口；
2. 向量模型离线模式：auto / online / offline 三种模式，离线时使用本地
   SentenceTransformer 模型，不发起任何外网请求；
3. 客户端复用：Chroma / OpenAI / LLM 客户端都做成进程内单例（由工厂负责）；
4. 文件重复上传：上传登记表 + SHA-256 内容指纹，实现“秒传”和覆盖导入。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# 读取项目根目录的 .env（不存在时静默跳过，不影响使用系统环境变量）
load_dotenv()

# 项目根目录（src/api/config.py 往上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 环境变量里常用的“真值”写法，统一转换成 bool
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _env_bool(name: str, default: bool = False) -> bool:
    """读取布尔型环境变量；未配置时返回默认值。"""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int) -> int:
    """读取整型环境变量；非法值回退默认值，避免启动直接失败。"""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """读取浮点型环境变量；非法值回退默认值。"""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(str(raw).strip())
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    """读取逗号分隔的列表型环境变量。"""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return list(default)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


class Settings:
    """API 配置类。

    说明：这里使用类属性而不是实例属性，是为了保持项目原有写法
    （``from src.api.config import settings`` 后直接访问），同时让配置在
    进程启动时只读取一次，避免每个请求都去查环境变量。
    """

    # ------------------------------------------------------------------
    # 基础 / 文件上传
    # ------------------------------------------------------------------
    MAX_FILE_SIZE = _env_int("MAX_FILE_SIZE", 50 * 1024 * 1024)  # 最大 50MB
    ALLOWED_FILE_EXTENSIONS = ['.pdf', '.docx', '.htm', '.html', '.md']
    # 上传文件存储目录（绝对路径，避免依赖启动时的工作目录）
    UPLOAD_DIR = str(PROJECT_ROOT / 'data' / 'uploads')
    # 向量库持久化目录
    CHROMA_PERSIST_DIR = str(PROJECT_ROOT / 'data' / 'chroma_db')
    # 向量相似度空间：cosine / l2 / ip
    CHROMA_DISTANCE = os.getenv("CHROMA_DISTANCE", "cosine")

    # ------------------------------------------------------------------
    # 向量模型（在线 / 离线 / 自动）
    # ------------------------------------------------------------------
    # auto：有在线密钥且未强制离线时走在线接口，否则走本地离线模型
    # online：强制使用 OpenAI 兼容的向量接口（硅基流动、百炼、OpenAI 等）
    # offline：强制使用本地 SentenceTransformer 模型，全程不联网
    EMBEDDING_MODE = (os.getenv("EMBEDDING_MODE") or "auto").strip().lower()
    # 在线向量模型配置；密钥优先 EMBEDDING_API_KEY，回退 SILICONFLOW_API_KEY
    EMBEDDING_API_KEY = (
        os.getenv("EMBEDDING_API_KEY") or os.getenv("SILICONFLOW_API_KEY") or ""
    )
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "BAAI/bge-m3"
    EMBEDDING_BASE_URL = (
        os.getenv("EMBEDDING_BASE_URL") or "https://api.siliconflow.cn/v1"
    )
    # 显式指定向量维度；不设置时按模型默认维度（OpenAIEmbeddings 内置映射）
    EMBEDDING_DIMENSION = _env_int("EMBEDDING_DIMENSION", 0) or None
    # 离线模型：默认使用本机已缓存的轻量中文模型，避免首次启动下载大模型
    OFFLINE_EMBEDDING_MODEL = (
        os.getenv("OFFLINE_EMBEDDING_MODEL") or "BAAI/bge-small-zh"
    )
    # 也可以直接指向本地模型目录（优先级高于 OFFLINE_EMBEDDING_MODEL）
    OFFLINE_EMBEDDING_DIR = os.getenv("OFFLINE_EMBEDDING_DIR") or ""
    OFFLINE_EMBEDDING_DEVICE = os.getenv("OFFLINE_EMBEDDING_DEVICE") or "cpu"
    # 离线加载时强制只读本地缓存，禁止 HuggingFace 访问网络
    OFFLINE_EMBEDDING_LOCAL_ONLY = _env_bool("OFFLINE_EMBEDDING_LOCAL_ONLY", True)
    # 离线模型的向量维度提示值：模型加载前用它做知识库兼容性检查。
    # BAAI/bge-small-zh = 512；换成其它模型时请同步修改，设为 0 表示未知。
    OFFLINE_EMBEDDING_DIMENSION = _env_int("OFFLINE_EMBEDDING_DIMENSION", 512) or None
    # 本地模型推理批大小（显存/内存紧张时调小）
    EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)
    # 服务启动时是否预热向量模型；默认关闭，避免离线大模型拖慢启动
    EMBEDDING_WARMUP_ON_STARTUP = _env_bool("EMBEDDING_WARMUP_ON_STARTUP", False)

    # ------------------------------------------------------------------
    # 向量缓存（内存 + 磁盘）
    # ------------------------------------------------------------------
    EMBEDDING_CACHE_ENABLED = _env_bool("EMBEDDING_CACHE_ENABLED", True)
    # 缓存数据库目录；SQLite 文件默认 data/cache/embeddings/embeddings.sqlite3
    EMBEDDING_CACHE_DIR = os.getenv("EMBEDDING_CACHE_DIR") or str(
        PROJECT_ROOT / "data" / "cache" / "embeddings"
    )
    # 内存 LRU 最多保存多少条向量（0 表示只用磁盘缓存）
    EMBEDDING_CACHE_MAX_ITEMS = _env_int("EMBEDDING_CACHE_MAX_ITEMS", 2048)
    # 磁盘缓存有效期（秒）；0 表示永不过期
    EMBEDDING_CACHE_TTL_SECONDS = _env_int("EMBEDDING_CACHE_TTL_SECONDS", 0)

    # ------------------------------------------------------------------
    # 文件重复上传 / 秒传
    # ------------------------------------------------------------------
    # 是否开启 SHA-256 内容去重；关闭后每次上传都会重新解析与向量化
    UPLOAD_DEDUP_ENABLED = _env_bool("UPLOAD_DEDUP_ENABLED", True)
    # 上传登记表（SQLite）：记录文件名、内容指纹、分块数、上传时间
    UPLOAD_REGISTRY_DB = os.getenv("UPLOAD_REGISTRY_DB") or str(
        PROJECT_ROOT / "data" / "upload_registry.sqlite3"
    )

    # ------------------------------------------------------------------
    # LLM 供应商配置
    # ------------------------------------------------------------------
    LLM_PROVIDER = {
        'zhipu': {
            'API_KEY': os.getenv('ZHIPU_API_KEY'),
            'BASE_URL': os.getenv('ZHIPU_BASE_URL'),
            'MODEL': os.getenv('ZHIPU_MODEL'),
        },
        'qwen': {
            'API_KEY': os.getenv('QWEN_API_KEY'),
            'BASE_URL': os.getenv('QWEN_BASE_URL'),
            'MODEL': os.getenv('QWEN_MODEL'),
        },
        'siliconflow': {
            'API_KEY': os.getenv('SILICONFLOW_API_KEY'),
            'BASE_URL': os.getenv('SILICONFLOW_BASE_URL'),
            'MODEL': os.getenv('SILICONFLOW_MODEL'),
        },
    }
    # LLM 参数
    LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.7)
    LLM_MAX_TOKENS = _env_int("LLM_MAX_TOKENS", 2000)
    LLM_TIMEOUT_SECONDS = _env_float("LLM_TIMEOUT_SECONDS", 120.0)
    TOP_K = _env_int("TOP_K", 5)
    Default_LLM_MODEL = os.getenv('DEFAULT_LLM_MODEL')
    Default_LLM_BASE_URL = os.getenv('DEFAULT_LLM_BASE_URL')
    Default_LLM_API_KEY = os.getenv('DEFAULT_LLM_API_KEY')
    # 与 Default_LLM_MODEL 等价的别名，供新代码使用
    DEFAULT_LLM_MODEL = Default_LLM_MODEL

    # ------------------------------------------------------------------
    # 前端 / CORS
    # ------------------------------------------------------------------
    # 前端构建产物目录；存在 index.html 时由 FastAPI 直接托管
    FRONTEND_DIST = os.getenv("FRONTEND_DIST") or str(
        PROJECT_ROOT / "frontend" / "dist"
    )
    # 开发模式下 Vite 默认端口；允许跨域访问后端
    CORS_ORIGINS = _env_list(
        "CORS_ORIGINS",
        [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
    )


settings = Settings()
