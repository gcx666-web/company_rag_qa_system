# -*- coding: utf-8 -*-
"""
main.py —— FastAPI 应用入口。

职责：

1. 注册健康检查、文件上传、知识库管理、对话、系统状态五组路由；
2. 配置 CORS，方便 Vite 开发服务器（5173）直接联调；
3. 如果前端已构建（``frontend/dist/index.html`` 存在），直接由 FastAPI 托管
   静态资源，并为单页应用提供 history 路由回退；
4. 启动时创建必要目录，并按配置决定是否预热向量模型。
"""

from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# 允许以 `python src/api/main.py` 方式从任意目录启动
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.api.config import settings
from src.api.routers import (
    chat_router,
    document_router,
    health_router,
    system_router,
    upload_file_router,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# 前端构建产物目录；不存在时后端只提供 API 与欢迎页
FRONTEND_DIST = Path(settings.FRONTEND_DIST)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建目录、打印关键配置，可选预热向量模型。"""
    for directory in (
        settings.UPLOAD_DIR,
        settings.CHROMA_PERSIST_DIR,
        settings.EMBEDDING_CACHE_DIR,
        str(Path(settings.UPLOAD_REGISTRY_DB).parent),
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)

    logger.info("RAG 服务启动：前端目录=%s", FRONTEND_DIST)
    logger.info(
        "向量模式=%s（配置=%s），模型=%s，缓存=%s",
        settings.EMBEDDING_MODE,
        settings.EMBEDDING_MODE,
        settings.OFFLINE_EMBEDDING_MODEL
        if settings.EMBEDDING_MODE == "offline"
        else settings.EMBEDDING_MODEL,
        settings.EMBEDDING_CACHE_ENABLED,
    )

    # 预热是可选项：默认关闭，避免离线大模型拖慢开发环境启动
    if settings.EMBEDDING_WARMUP_ON_STARTUP:
        from starlette.concurrency import run_in_threadpool
        from src.embeddings.factory import warmup_embedding

        try:
            result = await run_in_threadpool(warmup_embedding)
            logger.info(
                "向量模型预热完成：model=%s, dimension=%s, %.1f ms",
                result.get("model"),
                result.get("dimension"),
                result.get("elapsed_ms", 0.0),
            )
        except Exception as exc:  # 预热失败不阻止服务启动，由系统页提示用户
            logger.warning("向量模型预热失败（服务继续启动）：%s", exc)

    yield


app = FastAPI(
    title="RAG System FastAPI",
    description=(
        "公司知识库 RAG 问答系统：Vue 前端 + FastAPI 后端，"
        "支持向量缓存、离线向量模型、客户端复用与文件秒传。"
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# CORS：Vite 开发服务器默认 5173，预览服务器 4173
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由统一挂到 /api 前缀
app.include_router(health_router.router, prefix="/api")
app.include_router(upload_file_router.router, prefix="/api")
app.include_router(document_router.router, prefix="/api")
app.include_router(chat_router.router, prefix="/api")
app.include_router(system_router.router, prefix="/api")


@app.get("/api", tags=["根路径"])
async def api_index():
    """API 欢迎页（前端可直接访问 /api 检查服务是否启动）。"""
    return {
        "message": "Welcome to the RAG System FastAPI!",
        "docs": "/docs",
        "frontend": "/" if (FRONTEND_DIST / "index.html").is_file() else None,
    }


# ======================================================================
# 前端静态资源托管（生产模式：pnpm build 之后）
# ======================================================================
_INDEX_FILE = FRONTEND_DIST / "index.html"
if _INDEX_FILE.is_file():
    # Vite 默认把 JS/CSS 输出到 assets/，这里单独挂载以获得正确的 MIME 与缓存
    _ASSETS_DIR = FRONTEND_DIST / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        """单页应用回退：静态文件存在则返回文件，否则返回 index.html。

        - ``/api/...`` 的未知路径仍返回 404，避免把 API 404 变成 HTML；
        - 用 resolve + is_relative_to 阻止 ``../`` 读取前端目录之外的文件。
        """
        if full_path == "api" or full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")

        candidate = (FRONTEND_DIST / full_path).resolve()
        try:
            candidate.relative_to(FRONTEND_DIST.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="Not Found")
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_INDEX_FILE)
else:
    @app.get("/", tags=["根路径"])
    async def index():
        """前端尚未构建时的欢迎页。"""
        return {
            "message": "Welcome to the RAG System FastAPI!",
            "frontend": "前端尚未构建，请在 frontend/ 目录执行 pnpm install && pnpm build",
            "docs": "/docs",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="127.0.0.1", port=8000, reload=True)
