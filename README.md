# 企业知识库 RAG 问答系统

基于 **FastAPI + Vue 3 + ChromaDB + 大模型** 的企业知识库问答系统：上传文档后自动解析、切分、向量化，提问时先检索相关片段，再由大模型生成有依据的回答，并展示可溯源的引用来源。

本次升级重点补齐了四类工程优化（全部带中文注释）：

| 优化 | 说明 |
| --- | --- |
| 向量结果缓存 | 内存 LRU + SQLite 持久化，重复文本直接复用向量，进程重启后缓存仍有效 |
| 向量模型离线模式 | `auto / online / offline` 三种模式；离线时使用本地 SentenceTransformer，不访问外网 |
| 客户端复用 | Chroma、OpenAI、LLM、文档解析器按进程/线程复用，避免每个请求重复初始化 |
| 文件重复上传 | SHA-256 内容指纹 + 上传登记表，实现“秒传”、同名覆盖与并发保护 |

---

## 一、界面预览

前端包含四个页面（`frontend/src/views/`）：

| 页面 | 能力 |
| --- | --- |
| **对话问答** | 选择知识库与大模型、调整召回数量、SSE 流式逐字输出、Markdown 回答、引用来源卡片、耗时与 token 统计 |
| **文档上传** | 拖拽多文件、上传队列与进度、创建知识库、秒传/覆盖结果提示、强制重新导入 |
| **知识库** | 集合列表、文件列表、分块/表格/页数/大小/指纹、删除文件或整个知识库 |
| **系统状态** | 向量模式与模型、缓存命中率、四项优化开关、LLM 供应商、存储路径、预热/清空缓存 |

设计上使用深靛蓝侧栏 + 浅色卡片内容区，无 `vue-router`，通过 `KeepAlive` 保留对话记录与上传队列；移动端侧栏自动变为顶部导航。

---

## 二、技术栈

- **后端**：Python 3.13、FastAPI、ChromaDB、OpenAI SDK、sentence-transformers、Docling、Pydantic v2
- **前端**：Vue 3（`<script setup>`）、Vite、marked、DOMPurify
- **存储**：Chroma 持久化目录、SQLite（向量缓存 + 上传登记表）、本地上传目录

---

## 三、快速开始

### 1. 安装后端依赖

```bash
uv sync
```

或使用已有虚拟环境：

```bash
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS
```

### 2. 配置环境变量

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # Linux / macOS
```

至少配置一个大模型供应商的 `API_KEY`（如 `ZHIPU_API_KEY`）。向量模型可按需选择在线或离线，见下文。

### 3. 构建前端

```bash
cd frontend
pnpm install
pnpm build
cd ..
```

构建产物在 `frontend/dist`，FastAPI 启动时会自动托管；未构建时访问 `/` 会返回提示 JSON，API 仍可正常使用。

### 4. 启动服务

```bash
uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/> 即可使用；接口文档在 <http://127.0.0.1:8000/docs>。

### 5. 前端开发模式（可选）

后端保持 8000 端口运行，另开终端：

```bash
cd frontend
pnpm dev
```

访问 <http://127.0.0.1:5173/>，Vite 会把 `/api` 代理到 `127.0.0.1:8000`。

---

## 四、向量模式与离线运行

### 模式选择

```env
EMBEDDING_MODE=auto      # 有在线密钥走在线，否则离线
EMBEDDING_MODE=online    # 强制在线，缺少密钥会报错
EMBEDDING_MODE=offline   # 强制离线，使用本地模型
```

`auto` 的判定顺序：

1. `EMBEDDING_OFFLINE=1` → 离线；
2. 配置了 `EMBEDDING_API_KEY` 或 `SILICONFLOW_API_KEY` → 在线；
3. 否则 → 离线。

### 离线模型准备

默认离线模型为 `BAAI/bge-small-zh`（512 维）。首次使用前需要让模型存在于本机缓存：

```bash
# 联网环境下载一次（或直接从已有机器拷贝 HuggingFace 缓存）
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh')"
```

随后设置：

```env
EMBEDDING_MODE=offline
OFFLINE_EMBEDDING_MODEL=BAAI/bge-small-zh
OFFLINE_EMBEDDING_LOCAL_ONLY=true
OFFLINE_EMBEDDING_DIMENSION=512
```

离线模式会在加载模型前设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、`HF_DATASETS_OFFLINE=1`，运行期不会发起外网请求。模型采用懒加载：第一次上传或问答时才加载，避免拖慢服务启动；也可以在「系统状态」页手动“预热向量模型”。

> **注意：向量维度不可混用。** Chroma 集合一旦写入向量，维度就固定了。在线模型（如 `bge-m3` 1024 维）建立的旧知识库，在切到离线 512 维模型后无法检索或新增文档。前端会显示“维度不匹配”提示；处理方式是新建知识库重新上传，或切回原向量模型。离线模型换型时请同步修改 `OFFLINE_EMBEDDING_DIMENSION`。

---

## 五、四项优化的实现位置

### 1. 向量缓存

- `src/embeddings/embedding_cache.py`
  - `EmbeddingCache`：内存 LRU + SQLite（WAL），key 为 `模型名 + 文本` 的哈希；
  - `CachedEmbeddings`：包装任意 `Base` 向量模型，批量查询、只对未命中文本调用底层模型；
  - 缓存统计：命中/未命中、内存条数、磁盘条数、磁盘占用、命中率。
- `src/embeddings/factory.py` 统一给在线/离线模型套上缓存。
- 缓存文件默认 `data/cache/embeddings/embeddings.sqlite3`。

### 2. 离线模式

- `src/embeddings/local_embeddings.py`：懒加载、线程安全、离线环境变量、`local_files_only`。
- `src/embeddings/factory.py`：`resolve_embedding_mode()` 解析模式并做进程级单例。
- `EMBEDDING_WARMUP_ON_STARTUP` 可控制启动预热。

### 3. 客户端复用

- `src/database/chroma_client.py`：按持久化目录缓存 `PersistentClient`。
- `src/api/desp.py`：按集合缓存 `DocumentStore` / `SearchService`，按供应商缓存 `LLM`。
- `src/embeddings/factory.py`：按模式/模型/维度缓存向量模型。
- `src/parsers/__init__.py`：按扩展名在线程内缓存解析器实例。

### 4. 文件秒传与覆盖

- `src/database/upload_registry.py`：SQLite 登记表（集合、文件名、SHA-256、分块统计、状态）+ 上传锁池。
- `src/api/routers/upload_file_router.py`：
  - `created`：首次导入；
  - `replaced`：同名文件内容变化，删除旧分块后写入新分块；
  - `duplicate`：相同内容已成功入库，直接返回上次统计，**不解析、不向量化**；
  - `force=true`：忽略内容去重，强制重新导入（已缓存向量仍会复用）。
- `src/api/utils.py`：原子写文件、文件名清洗、SHA-256、维度错误提示。

---

## 六、主要 API

所有接口以 `/api` 为前缀：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/health/` | 健康检查 |
| POST | `/api/document/upload/?collection_name=&force=` | 上传文件（multipart 字段 `file`） |
| GET | `/api/document/collections` | 集合列表（含真实维度与兼容性） |
| POST | `/api/document/collections` | 创建集合，body `{"name": "hr-policy"}` |
| DELETE | `/api/document/collections/{name}` | 删除集合 |
| GET | `/api/document/files?collection_name=` | 文件列表 |
| DELETE | `/api/document/files?collection_name=&file_name=` | 删除文件分块 |
| GET | `/api/document/stats?collection_name=` | 集合统计 |
| GET/POST | `/api/document/support-types` | 支持的文件类型 |
| POST | `/api/chat/` | RAG 问答，body `{question, model, collection_name, top_k, max_tokens}` |
| POST | `/api/chat/stream` | RAG 问答（SSE 流式），请求体与 `/api/chat/` 相同 |
| GET | `/api/system/status` | 系统总览 |
| GET | `/api/system/embedding/status` | 向量模型状态 |
| GET | `/api/system/embedding/cache` | 缓存统计 |
| POST | `/api/system/embedding/warmup` | 预热向量模型 |
| POST | `/api/system/embedding/cache/clear` | 清空向量缓存 |

上传响应中的 `status` 为 `created / replaced / duplicate`；`embedding_cache_hits` 表示本次请求新增的缓存命中次数。

### 流式问答（SSE）

`POST /api/chat/stream` 在检索完成后以 `text/event-stream` 逐块返回答案，前端「对话问答」页默认使用该接口。事件如下：

| 事件 | data 字段 | 说明 |
| --- | --- | --- |
| `meta` | `question / context / sources / model / collection_name / retrieval_time_ms` | 流开始时的上下文、来源卡片与检索耗时 |
| `delta` | `content` | 增量文本，前端直接拼接即可逐字输出 |
| `done` | `answer / finish_reason / tokens_used / total_time_ms / retrieval_time_ms / model / collection_name` | 生成结束的汇总信息 |
| `error` | `message` | 流式过程中的错误（此时 HTTP 状态已是 200） |

响应头显式禁用各级缓存，避免浏览器或反向代理把响应攒批后一次性下发：

```text
Cache-Control: no-cache, no-store, must-revalidate, max-age=0
Pragma: no-cache
Expires: 0
X-Accel-Buffering: no
```

模型不可用、集合非法、向量维度不一致等问题会在流开始前返回 400 / 409 / 500；已经进入流式后发生的上游错误通过 `event: error` 推送，并保留已经生成的部分内容。客户端断开时服务端会停止生成并关闭上游连接。

调试示例（`-N` 关闭 curl 自身缓冲，可看到逐块输出）：

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"公司年假有多少天？","model":"zhipu","collection_name":"default","top_k":5}'
```

> `tokens_used` 依赖供应商支持 OpenAI 的 `stream_options.include_usage`；不支持时服务端会自动去掉该参数重试，此时该字段为空对象，不影响答案输出。

---

## 七、目录结构

```text
company_rag_qa_system/
├─ src/
│  ├─ api/
│  │  ├─ main.py                 # FastAPI 入口 + 前端静态托管
│  │  ├─ config.py               # 配置中心
│  │  ├─ desp.py                 # 依赖注入：客户端/实例复用
│  │  ├─ models.py               # Pydantic 请求响应模型
│  │  ├─ utils.py                # 校验、SHA-256、原子写、错误提示
│  │  └─ routers/                # 上传、知识库、对话、系统、健康检查
│  ├─ database/
│  │  ├─ chroma_client.py        # Chroma 客户端缓存
│  │  ├─ document_store.py       # 文档入库/检索封装
│  │  └─ upload_registry.py      # 上传登记表 + 上传锁
│  ├─ embeddings/
│  │  ├─ embedding_cache.py      # 向量缓存
│  │  ├─ factory.py              # 向量模型工厂（模式/单例/缓存）
│  │  ├─ local_embeddings.py     # 离线 SentenceTransformer
│  │  └─ chorm_store.py          # Chroma 向量存储
│  ├─ llm/                       # LLM 客户端与 RAG 提示词
│  ├─ parsers/                   # PDF/DOCX/HTML/MD 解析
│  └─ retrieval/                 # 检索服务
├─ frontend/
│  ├─ src/
│  │  ├─ api/client.js           # 接口封装 + 上传进度
│  │  ├─ store.js                # 全局状态与 Toast
│  │  ├─ components/             # 侧栏、图标、来源卡片、Toast
│  │  ├─ views/                  # 四个页面
│  │  └─ styles/main.css         # 设计系统与响应式
│  └─ vite.config.js
├─ tests/                        # unittest 测试
├─ data/                         # 运行期数据（默认被 git 忽略）
├─ .env.example
└─ README.md
```

---

## 八、测试

项目使用标准库 `unittest`：

```bash
python -m unittest discover -s tests -v
```

只跑本次新增的缓存与登记表测试：

```bash
python -m unittest tests.test_embedding_cache tests.test_upload_registry -v
```

前端构建检查：

```bash
cd frontend
pnpm build
```

---

## 九、常见问题

**Q：页面能打开，但系统状态显示“后端未连接”？**
确认后端已启动，且前端通过同源地址（FastAPI 托管的 `/`）或 Vite 代理访问；直接双击 `dist/index.html` 无法访问 API。

**Q：离线模式启动报“加载向量模型失败”？**
说明本机缓存没有该模型。联网下载一次，或设置 `OFFLINE_EMBEDDING_DIR` 指向已有模型目录，并保持 `OFFLINE_EMBEDDING_LOCAL_ONLY=true`。

**Q：旧知识库问答报 409 / 维度不一致？**
在线模型与离线模型的向量维度不同，Chroma 集合维度不可修改。新建知识库重新上传文档，或切回原来的向量模型。

**Q：重复上传没有秒传？**
检查 `UPLOAD_DEDUP_ENABLED=true`；秒传要求内容 SHA-256 一致、登记状态为 `success`，且向量库中仍存在该文件分块。也可以在知识库中删除文件后重新上传。

**Q：想清空缓存但保留文档？**
在「系统状态」点击“清空向量缓存”，只删除 `data/cache/embeddings` 中的文本→向量缓存，不会删除 Chroma 中的文档向量。
