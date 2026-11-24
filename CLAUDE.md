# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

企业知识库 RAG 问答系统：FastAPI 后端 + Vue 3 前端 + ChromaDB + OpenAI 兼容大模型。上传文档后解析、切分、向量化并写入 Chroma；提问时检索相关分块，再调用大模型生成带引用来源的回答（同步与 SSE 流式两种）。

- Python 3.13，依赖用 `uv` 管理（`.venv` 已存在，解释器固定为 `.venv/Scripts/python.exe`）。
- 前端位于 `frontend/`，Vue 3 `<script setup>` + Vite，无 vue-router、无 Pinia。
- 代码与文档注释统一使用中文。
- `src/` 是平铺的命名空间包（没有 `company_rag_qa_system` 包），导入一律写 `from src.xxx import ...`，因此**项目根目录必须在 `sys.path` 上**。`pyproject.toml` 里 `[tool.uv] package = false` 是刻意设置的，不要改回打包模式。

## 常用命令

后端（项目根目录执行）：

```bash
uv sync                                             # 安装依赖
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
.venv/Scripts/python.exe -m uvicorn src.api.main:app --reload   # 用现有 venv 直接启动
```

前端：

```bash
cd frontend
pnpm install
pnpm build        # 产物到 frontend/dist，FastAPI 启动时自动托管
pnpm dev          # 5173 端口，/api 代理到 127.0.0.1:8000
```

测试（标准库 `unittest`，全部 mock，不联网）：

```bash
.venv/Scripts/python.exe -m unittest discover -s tests -v
.venv/Scripts/python.exe -m unittest tests.test_embedding_cache -v   # 单个测试模块
.venv/Scripts/python.exe -m unittest tests.test_chunker.TestChunkTextMethod.test_短文本只生成一块 -v   # 单个用例（用例名含中文）
```

`.env` 从 `.env.example` 复制，至少配置一个 LLM 供应商密钥；`.env`、`data/`、`frontend/dist/` 均被 git 忽略。

## 架构

### 两条主链路

**文档入库**（`src/api/routers/upload_file_router.py`）：

```
UploadFile → safe_filename / _validate_upload → SHA-256
  → UploadRegistry 去重（秒传）
  → ParserFactory（thread-local 缓存 Docling 解析器）
  → ParseResult（base.py 的 chunks: type/content/page/chunk_id/metadata）
  → DocumentStore.add_parse_result
       ├─ overwrite=True 时先 delete_by_file_name 删除旧分块
       ├─ 写入前 _ensure_ids_absent 拦截已存在 id
       └─ embed_batch → ChromaVectorStore.add_documents
```

**问答**（`src/api/routers/chat_router.py`）：

```
model → _resolve_provider（供应商名或模型名 → zhipu/qwen/siliconflow）
  → get_search_service(collection) → SearchService.search
       → DocumentStore.search → embed_text → Chroma search → 结果字典
  → format_context + build_prompt（src/llm/rag.py）
  → LLM.generate（同步）/ LLM.generate_stream（SSE）
```

`POST /api/chat/stream` 在检索完成后推送 SSE：`meta`（上下文/来源/检索耗时）→ 多次 `delta` → `done`，流内错误用 `event: error`。响应头显式禁用缓存与 `X-Accel-Buffering`，否则流式会被浏览器/代理攒批。

### 关键设计约束

- **配置中心**：所有可调参数集中在 [src/api/config.py](src/api/config.py) 的 `Settings` 类属性，导入时只读一次环境变量。新增配置在 `.env.example` 和该文件同步添加，不要在业务代码里散落 `os.getenv`。
- **进程内单例 / 复用**：依赖注入在 [src/api/desp.py](src/api/desp.py)——Chroma 客户端按持久化目录、`DocumentStore`/`SearchService` 按集合名、`LLM` 按供应商+模型+地址缓存；向量模型由 [src/embeddings/factory.py](src/embeddings/factory.py) 按模式/模型/维度缓存。缓存操作都带 `RLock`。删除集合后需调用 `evict_document_store`。
- **namespace = 集合名**：分块写入 metadata 的 `namespace`，检索按它过滤。文档 id 是确定性的 `{namespace}_{file_name}_{chunk_id}`；纯文本写入用 `{namespace}_{uuid4.hex}`。
- **Chroma 静默忽略重复 id**：`collection.add` 对已存在 id 不报错也不覆盖。因此 `DocumentStore` 写入前显式检查并抛 `DuplicateIDError`；覆盖导入必须先删旧分块。
- **向量维度不可混用**：Chroma 集合维度首次写入后固定。在线模型（如 bge-m3 1024 维）与离线模型（如 bge-small-zh 512 维）不能共用旧集合；遇到 Chroma 维度报错时用 `src/api/utils.py:embedding_dimension_hint` 翻译成 409 + 中文处理建议。
- **向量缓存**：`EmbeddingCache`（内存 LRU + SQLite WAL）+ `CachedEmbeddings` 包装器，key 为「模型名 + 文本」哈希。缓存文件在 `data/cache/embeddings/`。
- **上传去重**：`UploadRegistry`（SQLite，`UNIQUE(collection, file_name)`）+ SHA-256 指纹，状态 `success/processing/failed`；`UploadLockPool` 对「集合+内容」和「集合+文件名」双键加锁。上传结果 `status` 为 `created / replaced / duplicate`。
- **阻塞操作放线程池**：Docling 解析、Chroma 查询、离线模型加载在路由里用 `run_in_threadpool`，避免阻塞事件循环。
- **错误语义**：非法集合名/参数 400，重复处理中或维度不匹配 409，文件过大 413，检索失败 500，LLM 上游失败 502，集合初始化失败 503；错误信息要可操作的中文。

### 前端结构

- [frontend/src/App.vue](frontend/src/App.vue) 用 `view` ref + 组件映射切换四个页面，`<KeepAlive>` 保留对话记录与上传队列；新增页面需要同时改 `views` 映射与 [frontend/src/components/SidebarNav.vue](frontend/src/components/SidebarNav.vue)。
- 全局状态在 [frontend/src/store.js](frontend/src/store.js)（Vue `reactive`），接口封装在 [frontend/src/api/client.js](frontend/src/api/client.js)：普通请求用 fetch，上传用 XHR 以获取进度；`ApiError` 统一提取 FastAPI 的 `detail`。
- SSE 由 [frontend/src/views/ChatView.vue](frontend/src/views/ChatView.vue) 消费；Markdown 渲染走 `marked` + `DOMPurify`。
- `frontend/dist` 被忽略但运行期需要；没有构建产物时访问 `/` 返回提示 JSON，API 不受影响。

### 目录职责

- `src/api/`：入口、配置、模型、依赖注入、路由（upload/document/chat/system/health）。
- `src/parsers/`：`ParserFactory` 按扩展名分派（pdf/docx/md/html/htm），Docling 解析 + 清洗 + 文本/表格分开分块。
- `src/embeddings/`：`Base` 抽象（embed_text/embed_batch/get_dimension/get_model_name）、在线 OpenAI 兼容模型、离线 SentenceTransformer、缓存、工厂、Chroma 实现。
- `src/database/`：Chroma 客户端缓存、`DocumentStore` 编排、上传登记表。
- `src/llm/`：`LLM` 同步/异步流式客户端、`rag.py` 上下文格式化与提示词。
- `src/retrieval/`：`SearchService` + `RetrievalConfig`（top_k/min_score/namespace）。
- `tests/`：unittest，测试文件自行把项目根目录插入 `sys.path`；新增测试沿用该模式。

## 修改时的注意事项

- Chroma metadata 只接受标量或同类型标量列表；`None` 与嵌套结构会被 `ChromaVectorStore` 丢弃。
- 流式接口保持「校验与检索先同步完成，再进入 SSE」的顺序，这样参数/维度错误仍能返回 400/409。
- 改向量模型或维度时，同步更新 `OFFLINE_EMBEDDING_DIMENSION`、前端兼容性提示与缓存 key 逻辑。
- 新增文件类型需同时改 `ParserFactory._parsers`、`Settings.ALLOWED_FILE_EXTENSIONS` 及前端兜底后缀列表。
- 运行期数据在 `data/`（Chroma、上传文件、缓存、登记表），测试不要写入真实目录，使用临时目录并清理。
