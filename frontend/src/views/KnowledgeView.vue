<script setup>
/**
 * KnowledgeView —— 知识库管理页面。
 *
 * 左侧是集合（知识库）列表，右侧是当前集合的文件列表：
 * - 文件按 file_name 聚合展示分块数、表格数、页数、大小与上传状态；
 * - 支持删除单个文件（清理全部分块与登记记录）和删除整个集合；
 * - 所有数据来自 /api/document/*，不直接读 Chroma 文件。
 */
import { computed, onMounted, ref, watch } from 'vue'
import { api } from '../api/client'
import { notify, refreshCollections, refreshSystem, state } from '../store'
import { formatBytes, formatNumber, formatTime, timeAgo } from '../utils/format'
import AppIcon from '../components/AppIcon.vue'

const selected = ref('')
const files = ref([])
const loadingFiles = ref(false)
const filesError = ref('')
const newName = ref('')
const creating = ref(false)
const deletingCollection = ref(false)
const deletingFile = ref('')

const selectedInfo = computed(() =>
  state.collections.find((item) => item.name === selected.value),
)

const totals = computed(() =>
  state.collections.reduce(
    (acc, item) => {
      acc.chunks += Number(item.count || 0)
      acc.files += Number(item.files || 0)
      return acc
    },
    { chunks: 0, files: 0 },
  ),
)

watch(
  () => state.collections,
  (collections) => {
    if (!collections.length) {
      selected.value = ''
      files.value = []
      return
    }
    if (!collections.some((item) => item.name === selected.value)) {
      selected.value =
        collections.find((item) => item.name === 'default')?.name || collections[0].name
    }
  },
  { immediate: true, deep: true },
)

watch(selected, (name) => {
  if (name) loadFiles(name)
  else files.value = []
})

onMounted(() => {
  refreshCollections({ silent: true }).then(() => {
    if (selected.value) loadFiles(selected.value)
  })
  if (!state.system) refreshSystem({ silent: true })
})

/** 加载当前集合的文件列表。 */
async function loadFiles(name = selected.value) {
  if (!name) return
  loadingFiles.value = true
  filesError.value = ''
  try {
    const payload = await api.listFiles(name)
    // 防止快速切换集合时，旧请求覆盖新集合的数据
    if (name === selected.value) files.value = payload?.files || []
  } catch (error) {
    filesError.value = error.message || '文件列表加载失败'
    files.value = []
  } finally {
    loadingFiles.value = false
  }
}

/** 选择集合。 */
function selectCollection(name) {
  selected.value = name
}

/** 创建集合。 */
async function createCollection() {
  const name = newName.value.trim()
  if (!name) {
    notify('请输入知识库名称', 'warning')
    return
  }
  creating.value = true
  try {
    const info = await api.createCollection(name)
    notify(`知识库「${info.name}」已创建`, 'success')
    newName.value = ''
    await refreshCollections({ silent: true })
    selected.value = info.name
  } catch (error) {
    notify(error.message, 'error')
  } finally {
    creating.value = false
  }
}

/** 删除整个集合（危险操作，二次确认）。 */
async function removeCollection() {
  if (!selected.value) return
  const info = selectedInfo.value
  const confirmed = window.confirm(
    `确定删除知识库「${selected.value}」吗？\n该操作会删除其中 ${info?.count || 0} 个分块，且不可恢复。`,
  )
  if (!confirmed) return
  deletingCollection.value = true
  try {
    const result = await api.deleteCollection(selected.value)
    notify(result.deleted ? '知识库已删除' : '知识库删除失败', result.deleted ? 'success' : 'error')
    selected.value = ''
    files.value = []
    await refreshCollections({ silent: true })
    refreshSystem({ silent: true })
  } catch (error) {
    notify(error.message, 'error')
  } finally {
    deletingCollection.value = false
  }
}

/** 删除单个文件的全部分块。 */
async function removeFile(file) {
  const confirmed = window.confirm(
    `确定删除文件「${file.file_name}」吗？\n将删除 ${file.chunks} 个分块，且不可恢复。`,
  )
  if (!confirmed) return
  deletingFile.value = file.file_name
  try {
    const result = await api.deleteFile(selected.value, file.file_name)
    notify(
      result.deleted ? `已删除 ${result.removed_chunks} 个分块` : '未找到可删除的分块',
      result.deleted ? 'success' : 'warning',
    )
    await Promise.all([loadFiles(), refreshCollections({ silent: true })])
    refreshSystem({ silent: true })
  } catch (error) {
    notify(error.message, 'error')
  } finally {
    deletingFile.value = ''
  }
}

/** 页码展示：最多 5 个，超出用 +N 表示。 */
function pagesText(pages) {
  if (!Array.isArray(pages) || !pages.length) return '—'
  const head = pages.slice(0, 5).join('、')
  return pages.length > 5 ? `${head} 等 ${pages.length} 页` : head
}

/** 登记状态对应的文案与颜色。 */
function statusLabel(file) {
  if (file.status === 'failed') return '导入失败'
  if (file.status === 'processing') return '处理中'
  return '已入库'
}
</script>

<template>
  <section class="view knowledge-view">
    <header class="page-head">
      <div>
        <p class="eyebrow">知识库 · 向量索引</p>
        <h1>知识库管理</h1>
        <p class="page-sub">查看集合与文件、删除失效文档，掌握每个知识库的索引规模。</p>
      </div>
      <div class="page-actions">
        <button class="btn btn-ghost" type="button" @click="loadFiles()">
          <AppIcon name="refresh" :size="16" />
          刷新文件
        </button>
      </div>
    </header>

    <div class="knowledge-stats">
      <div class="stat-card">
        <span class="stat-icon tone-indigo"><AppIcon name="database" :size="20" /></span>
        <div>
          <p class="stat-label">知识库</p>
          <p class="stat-value">{{ formatNumber(state.collections.length) }}</p>
        </div>
      </div>
      <div class="stat-card">
        <span class="stat-icon tone-emerald"><AppIcon name="layers" :size="20" /></span>
        <div>
          <p class="stat-label">文档数</p>
          <p class="stat-value">{{ formatNumber(totals.files) }}</p>
        </div>
      </div>
      <div class="stat-card">
        <span class="stat-icon tone-amber"><AppIcon name="zap" :size="20" /></span>
        <div>
          <p class="stat-label">分块总数</p>
          <p class="stat-value">{{ formatNumber(totals.chunks) }}</p>
        </div>
      </div>
      <div class="stat-card">
        <span class="stat-icon tone-slate"><AppIcon name="cpu" :size="20" /></span>
        <div>
          <p class="stat-label">向量模型</p>
          <p class="stat-value stat-value-sm">
            {{ state.system?.embedding?.model || '—' }}
          </p>
        </div>
      </div>
    </div>

    <div class="knowledge-layout">
      <aside class="card collection-panel">
        <header class="card-head compact">
          <div>
            <h3>集合列表</h3>
            <p>共 {{ state.collections.length }} 个知识库</p>
          </div>
          <button
            class="icon-btn"
            type="button"
            title="刷新列表"
            @click="refreshCollections({ silent: false })"
          >
            <AppIcon name="refresh" :size="16" />
          </button>
        </header>

        <div v-if="state.collectionsLoading" class="skeleton-list">
          <span v-for="n in 3" :key="n" class="skeleton-row" />
        </div>

        <ul v-else-if="state.collections.length" class="collection-list">
          <li
            v-for="item in state.collections"
            :key="item.name"
            class="collection-item"
            :class="{ active: item.name === selected }"
          >
            <button type="button" class="collection-main" @click="selectCollection(item.name)">
              <span class="collection-icon">
                <AppIcon name="database" :size="17" />
              </span>
              <span class="collection-copy">
                <strong>{{ item.name }}</strong>
                <small v-if="item.error" class="collection-error" :title="item.error">
                  读取失败 · 配置或向量库异常
                </small>
                <small v-else>
                  {{ item.count }} 分块 · {{ item.files }} 文件<template
                    v-if="item.embedding_dimension"
                  >
                    · {{ item.embedding_dimension }} 维</template
                  >
                </small>
              </span>
            </button>
            <button
              class="icon-btn danger"
              type="button"
              title="删除该知识库"
              :disabled="deletingCollection"
              @click="selected = item.name; removeCollection()"
            >
              <AppIcon name="trash" :size="15" />
            </button>
          </li>
        </ul>

        <div v-else class="panel-empty">
          <AppIcon name="database" :size="26" />
          <p>还没有知识库，先创建一个吧。</p>
        </div>

        <form class="create-collection" @submit.prevent="createCollection">
          <input
            v-model="newName"
            class="input"
            type="text"
            maxlength="64"
            placeholder="新知识库名称，如 hr-policy"
            :disabled="creating"
          />
          <button
            class="btn btn-secondary btn-sm"
            type="submit"
            :disabled="creating || !newName.trim()"
          >
            <AppIcon name="plus" :size="15" />
            {{ creating ? '创建中' : '创建' }}
          </button>
        </form>
      </aside>

      <div class="card file-panel">
        <header class="card-head file-head">
          <div>
            <h3>{{ selected || '文件列表' }}</h3>
            <p v-if="selectedInfo && selectedInfo.error" class="file-head-error">
              统计信息不可用 · 集合读取失败，原因见下方提示
            </p>
            <p v-else-if="selectedInfo">
              {{ selectedInfo.count }} 个分块 · {{ selectedInfo.files }} 个文件 ·
              距离 {{ selectedInfo.distance }}
            </p>
            <p v-else>请选择一个知识库查看文件</p>
          </div>
          <div class="queue-actions">
            <button
              class="btn btn-ghost btn-sm"
              type="button"
              :disabled="!selected || loadingFiles"
              @click="loadFiles()"
            >
              <AppIcon name="refresh" :size="15" />
              刷新
            </button>
            <button
              class="btn btn-danger btn-sm"
              type="button"
              :disabled="!selected || deletingCollection"
              @click="removeCollection"
            >
              <AppIcon name="trash" :size="15" />
              删除知识库
            </button>
          </div>
        </header>

        <div v-if="selectedInfo && selectedInfo.error" class="inline-alert error compat-banner">
          <AppIcon name="alert" :size="18" />
          <span>
            知识库「{{ selectedInfo.name }}」读取失败：{{ selectedInfo.error }}
            <template v-if="selectedInfo.model_dimension">
              （当前向量模型输出 {{ selectedInfo.model_dimension }} 维）
            </template>
          </span>
        </div>

        <div
          v-else-if="selectedInfo && selectedInfo.compatible === false"
          class="inline-alert error compat-banner"
        >
          <AppIcon name="alert" :size="18" />
          <span>
            该知识库使用 {{ selectedInfo.embedding_dimension }} 维向量，当前模型输出
            {{ selectedInfo.model_dimension }} 维，问答与新增上传都会被拒绝。
            请新建知识库重新导入文档，或切回原向量模型。
          </span>
        </div>

        <div v-if="loadingFiles" class="table-loading">
          <div v-for="n in 4" :key="n" class="skeleton-row" />
        </div>

        <div v-else-if="filesError" class="inline-alert error">
          <AppIcon name="alert" :size="18" />
          <span>{{ filesError }}</span>
        </div>

        <div v-else-if="files.length" class="file-table-wrap">
          <table class="file-table">
            <thead>
              <tr>
                <th>文件</th>
                <th>分块</th>
                <th>表格</th>
                <th>页数</th>
                <th>大小</th>
                <th>上传时间</th>
                <th>状态</th>
                <th class="cell-actions">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="file in files" :key="file.file_name">
                <td>
                  <div class="file-name-cell">
                    <span class="file-chip"><AppIcon name="file" :size="15" /></span>
                    <div>
                      <strong>{{ file.file_name }}</strong>
                      <small v-if="file.sha256">
                        指纹 {{ file.sha256.slice(0, 12) }}… · {{ formatNumber(file.total_chars) }} 字符
                      </small>
                    </div>
                  </div>
                </td>
                <td>{{ formatNumber(file.chunks) }}</td>
                <td>{{ formatNumber(file.tables) }}</td>
                <td class="cell-pages" :title="(file.pages || []).join('、')">
                  {{ pagesText(file.pages) }}
                </td>
                <td>{{ file.size ? formatBytes(file.size) : '—' }}</td>
                <td :title="formatTime(file.uploaded_at)">
                  {{ file.uploaded_at ? timeAgo(file.uploaded_at) : '—' }}
                </td>
                <td>
                  <span class="status-tag" :class="`status-${file.status || 'success'}`">
                    {{ statusLabel(file) }}
                  </span>
                </td>
                <td class="cell-actions">
                  <button
                    class="icon-btn danger"
                    type="button"
                    title="删除文件"
                    :disabled="deletingFile === file.file_name"
                    @click="removeFile(file)"
                  >
                    <AppIcon name="trash" :size="15" />
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <div v-else class="panel-empty tall">
          <AppIcon name="file" :size="30" />
          <p>{{ selected ? '这个知识库还没有文件' : '请选择左侧知识库' }}</p>
          <span v-if="selected">到「文档上传」导入 PDF、Word、HTML 或 Markdown 文档。</span>
        </div>
      </div>
    </div>
  </section>
</template>
