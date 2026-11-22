<script setup>
/**
 * UploadView —— 文档上传页面。
 *
 * 前端负责：选择/创建知识库、拖拽多文件、逐文件上传并展示进度；
 * 后端负责：格式/大小校验、SHA-256 秒传、覆盖导入、解析与向量化。
 *
 * 上传结果用四种状态展示：
 * - created   首次导入成功
 * - replaced  同名文件已更新（旧分块删除）
 * - duplicate 内容重复，后端直接秒传（没有重新解析 / 向量化）
 * - error     失败原因
 */
import { computed, onMounted, ref, watch } from 'vue'
import { api, uploadFile } from '../api/client'
import {
  notify,
  refreshCollections,
  refreshSystem,
  state,
} from '../store'
import { formatBytes, formatTime } from '../utils/format'
import AppIcon from '../components/AppIcon.vue'

const selectedCollection = ref('')
const newCollectionName = ref('')
const creating = ref(false)
const force = ref(false)
const dragActive = ref(false)
const queue = ref([])
const fileInput = ref(null)
const uploading = ref(false)
let itemSeed = 0

const acceptedExtensions = computed(() =>
  state.supportTypes.map((item) => item.toLowerCase()),
)

const selectedInfo = computed(() =>
  state.collections.find((item) => item.name === selectedCollection.value),
)

// 旧的空集合也可能继承了旧维度；compatible=false 时 Chroma 会拒绝新向量
const incompatible = computed(() => selectedInfo.value?.compatible === false)

const pendingCount = computed(
  () => queue.value.filter((item) => item.status === 'queued').length,
)

const finishedCount = computed(
  () =>
    queue.value.filter((item) =>
      ['created', 'replaced', 'duplicate', 'error'].includes(item.status),
    ).length,
)

const maxFileMb = 50
const statusText = {
  queued: '等待上传',
  uploading: '上传中',
  processing: '解析与向量化中',
  created: '导入成功',
  replaced: '已覆盖旧版本',
  duplicate: '秒传完成',
  error: '失败',
}

watch(
  () => state.collections,
  (collections) => {
    if (!collections.length) {
      selectedCollection.value = ''
      return
    }
    if (!collections.some((item) => item.name === selectedCollection.value)) {
      selectedCollection.value =
        collections.find((item) => item.name === 'default')?.name || collections[0].name
    }
  },
  { immediate: true, deep: true },
)

onMounted(() => {
  if (!state.collections.length) refreshCollections({ silent: true })
  if (!state.system) refreshSystem({ silent: true })
})

/** 打开系统文件选择框。 */
function pickFiles() {
  if (uploading.value) return
  fileInput.value?.click()
}

function onFileChange(event) {
  addFiles(Array.from(event.target.files || []))
  event.target.value = ''
}

function onDrop(event) {
  dragActive.value = false
  if (uploading.value) return
  addFiles(Array.from(event.dataTransfer?.files || []))
}

function onDragLeave(event) {
  // relatedTarget 为空或不在拖拽区内时才取消高亮，避免子元素触发闪烁
  if (!event.currentTarget.contains(event.relatedTarget)) dragActive.value = false
}

/** 客户端先做一次格式/大小预校验，减少无效上传。 */
function addFiles(files) {
  if (!files.length) return
  let added = 0
  for (const file of files) {
    const dot = file.name.lastIndexOf('.')
    const ext = dot >= 0 ? file.name.slice(dot).toLowerCase() : ''
    if (!acceptedExtensions.value.includes(ext)) {
      notify(`不支持的文件格式：${file.name}`, 'warning')
      continue
    }
    if (file.size > maxFileMb * 1024 * 1024) {
      notify(`${file.name} 超过 ${maxFileMb} MB 限制`, 'warning')
      continue
    }
    // 同一批次里避免重复加入完全相同的文件
    const duplicated = queue.value.some(
      (item) =>
        item.file.name === file.name &&
        item.file.size === file.size &&
        item.file.lastModified === file.lastModified,
    )
    if (duplicated) continue

    itemSeed += 1
    queue.value.push({
      id: itemSeed,
      file,
      name: file.name,
      size: file.size,
      status: 'queued',
      progress: 0,
      result: null,
      error: '',
      addedAt: Date.now() / 1000,
    })
    added += 1
  }
  if (added) notify(`已加入 ${added} 个文件，点击「开始上传」`, 'info', 2600)
}

/** 创建新集合。 */
async function createCollection() {
  const name = newCollectionName.value.trim()
  if (!name) {
    notify('请输入知识库名称', 'warning')
    return
  }
  creating.value = true
  try {
    const info = await api.createCollection(name)
    notify(`知识库「${info.name}」已创建`, 'success')
    newCollectionName.value = ''
    await refreshCollections({ silent: true })
    selectedCollection.value = info.name
    refreshSystem({ silent: true })
  } catch (error) {
    notify(error.message, 'error')
  } finally {
    creating.value = false
  }
}

/** 依次上传队列中的文件（串行，避免大量解析同时压垮后端）。 */
async function uploadAll() {
  if (!selectedCollection.value) {
    notify('请先选择或创建知识库', 'warning')
    return
  }
  if (incompatible.value) {
    notify(
      '当前知识库与向量模型维度不一致，请新建知识库后再上传',
      'warning',
      7000,
    )
    return
  }
  const targets = queue.value.filter((item) => item.status === 'queued')
  if (!targets.length) {
    notify('没有等待上传的文件', 'info')
    return
  }

  uploading.value = true
  const summary = { created: 0, replaced: 0, duplicate: 0, error: 0 }

  for (const item of targets) {
    item.status = 'uploading'
    item.progress = 0
    item.error = ''
    item.result = null
    try {
      const result = await uploadFile({
        file: item.file,
        collectionName: selectedCollection.value,
        force: force.value,
        onProgress: (ratio) => {
          item.progress = ratio
          // 上传字节完成后，后端仍在解析/向量化；把状态切成“处理中”
          if (ratio >= 1) item.status = 'processing'
        },
      })
      item.result = result
      item.progress = 1
      item.status = result?.status || 'created'
      if (summary[item.status] !== undefined) summary[item.status] += 1
    } catch (error) {
      item.status = 'error'
      item.error = error.message || '上传失败'
      summary.error += 1
    }
  }

  uploading.value = false
  await refreshCollections({ silent: true })
  refreshSystem({ silent: true })

  if (summary.error) {
    notify(
      `上传完成：成功 ${summary.created + summary.replaced}，秒传 ${summary.duplicate}，失败 ${summary.error}`,
      'warning',
      6000,
    )
  } else {
    notify(
      `上传完成：新建 ${summary.created}，覆盖 ${summary.replaced}，秒传 ${summary.duplicate}`,
      'success',
      6000,
    )
  }
}

/** 从队列移除（上传中的任务不允许移除）。 */
function removeItem(item) {
  if (['uploading', 'processing'].includes(item.status)) return
  const index = queue.value.findIndex((entry) => entry.id === item.id)
  if (index >= 0) queue.value.splice(index, 1)
}

/** 清空已完成/失败的任务。 */
function clearFinished() {
  queue.value = queue.value.filter(
    (item) => !['created', 'replaced', 'duplicate', 'error'].includes(item.status),
  )
}

/** 失败任务重新排队。 */
function retryItem(item) {
  if (uploading.value) return
  item.status = 'queued'
  item.progress = 0
  item.error = ''
  uploadAll()
}

/** 结果统计文案。 */
function resultText(item) {
  const result = item.result
  if (!result) return ''
  if (item.status === 'duplicate') {
    const other = result.existing_file_name
      ? `（相同内容已存在于「${result.existing_file_name}」）`
      : ''
    return `未重复解析与向量化${other} · 原有 ${result.chunks_count} 个分块`
  }
  const parts = [
    `${result.total_pages || 0} 页`,
    `${result.chunks_count || 0} 个分块`,
    `${result.table_count || 0} 个表格`,
  ]
  if (result.embedding_cache_hits > 0) {
    parts.push(`复用 ${result.embedding_cache_hits} 条缓存向量`)
  }
  return parts.join(' · ')
}

/** 状态对应的图标。 */
function statusIcon(status) {
  if (status === 'created' || status === 'replaced' || status === 'duplicate') return 'check'
  if (status === 'error') return 'alert'
  if (status === 'processing') return 'cpu'
  return 'upload'
}
</script>

<template>
  <section class="view upload-view">
    <header class="page-head">
      <div>
        <p class="eyebrow">文档接入 · 自动去重</p>
        <h1>文档上传</h1>
        <p class="page-sub">
          支持 PDF / DOCX / HTML / Markdown；相同内容会通过 SHA-256 指纹自动秒传。
        </p>
      </div>
      <div class="page-actions">
        <button class="btn btn-ghost" type="button" @click="refreshCollections({ silent: false })">
          <AppIcon name="refresh" :size="16" />
          刷新知识库
        </button>
      </div>
    </header>

    <div class="upload-grid">
      <div class="upload-main">
        <!-- 目标知识库与选项 -->
        <div class="card target-card">
          <div class="target-row">
            <label class="field target-select">
              <span class="field-label">导入到知识库</span>
              <select v-model="selectedCollection" :disabled="uploading" class="input select">
                <option value="" disabled>请选择知识库</option>
                <option v-for="item in state.collections" :key="item.name" :value="item.name">
                  {{ item.name }}（{{ item.count }} 个分块）
                </option>
              </select>
            </label>

            <div class="create-inline">
              <label class="field">
                <span class="field-label">新建知识库</span>
                <input
                  v-model="newCollectionName"
                  class="input"
                  type="text"
                  maxlength="64"
                  placeholder="例如 hr-policy"
                  :disabled="creating || uploading"
                  @keydown.enter.prevent="createCollection"
                />
              </label>
              <button
                class="btn btn-secondary"
                type="button"
                :disabled="creating || uploading || !newCollectionName.trim()"
                @click="createCollection"
              >
                <AppIcon name="plus" :size="16" />
                {{ creating ? '创建中…' : '创建' }}
              </button>
            </div>
          </div>

          <label class="check-row">
            <input v-model="force" type="checkbox" :disabled="uploading" />
            <span>
              <strong>强制重新导入</strong>
              忽略内容去重，重新解析并覆盖同名文件（已缓存的向量仍会复用）
            </span>
          </label>

          <div v-if="incompatible" class="inline-alert error compat-banner">
            <AppIcon name="alert" :size="18" />
            <span>
              「{{ selectedInfo.name }}」使用
              {{ selectedInfo.embedding_dimension }} 维向量，当前模型输出
              {{ selectedInfo.model_dimension }} 维。Chroma 集合维度不可修改，
              请新建知识库并重新上传文档。
            </span>
          </div>
        </div>

        <!-- 拖拽区域 -->
        <div
          class="dropzone"
          :class="{ active: dragActive, busy: uploading }"
          role="button"
          tabindex="0"
          @click="pickFiles"
          @keydown.enter.prevent="pickFiles"
          @keydown.space.prevent="pickFiles"
          @dragover.prevent
          @dragenter.prevent="dragActive = true"
          @dragleave="onDragLeave"
          @drop.prevent="onDrop"
        >
          <input
            ref="fileInput"
            class="visually-hidden"
            type="file"
            multiple
            :accept="acceptedExtensions.join(',')"
            @change="onFileChange"
          />
          <div class="dropzone-icon">
            <AppIcon name="upload" :size="30" />
          </div>
          <h2>{{ dragActive ? '松开鼠标，加入上传队列' : '拖拽文件到这里，或点击选择' }}</h2>
          <p>
            支持 {{ acceptedExtensions.join(' / ') }}，单个文件最大 {{ maxFileMb }} MB
          </p>
          <span class="dropzone-hint">上传前会计算文件指纹，重复内容不会再次向量化</span>
        </div>

        <!-- 上传队列 -->
        <div v-if="queue.length" class="card queue-card">
          <header class="card-head">
            <div>
              <h3>上传队列</h3>
              <p>{{ queue.length }} 个文件 · 待上传 {{ pendingCount }} · 已完成 {{ finishedCount }}</p>
            </div>
            <div class="queue-actions">
              <button
                class="btn btn-ghost btn-sm"
                type="button"
                :disabled="uploading || !finishedCount"
                @click="clearFinished"
              >
                清空已完成
              </button>
              <button
                class="btn btn-primary"
                type="button"
                :disabled="uploading || !pendingCount || incompatible"
                @click="uploadAll"
              >
                <AppIcon name="upload" :size="16" />
                {{ uploading ? '上传中…' : `开始上传${pendingCount ? `（${pendingCount}）` : ''}` }}
              </button>
            </div>
          </header>

          <ul class="queue-list">
            <li v-for="item in queue" :key="item.id" class="queue-item">
              <span class="queue-icon" :class="`status-${item.status}`">
                <AppIcon :name="statusIcon(item.status)" :size="18" />
              </span>

              <div class="queue-body">
                <div class="queue-title-row">
                  <strong class="queue-name" :title="item.name">{{ item.name }}</strong>
                  <span class="status-tag" :class="`status-${item.status}`">
                    {{ statusText[item.status] }}
                  </span>
                  <span class="queue-size">{{ formatBytes(item.size) }}</span>
                </div>

                <div
                  v-if="['uploading', 'processing'].includes(item.status)"
                  class="progress"
                  role="progressbar"
                  :aria-valuenow="Math.round(item.progress * 100)"
                  aria-valuemin="0"
                  aria-valuemax="100"
                >
                  <span :style="{ width: `${Math.max(4, item.progress * 100)}%` }" />
                </div>

                <p v-if="item.error" class="queue-error">{{ item.error }}</p>
                <p v-else-if="item.result" class="queue-result" :class="`status-${item.status}`">
                  {{ resultText(item) }}
                </p>
                <p v-else class="queue-muted">等待开始上传…</p>
              </div>

              <div class="queue-item-actions">
                <button
                  v-if="item.status === 'error'"
                  class="icon-btn"
                  type="button"
                  title="重试"
                  :disabled="uploading"
                  @click="retryItem(item)"
                >
                  <AppIcon name="refresh" :size="16" />
                </button>
                <button
                  v-if="!['uploading', 'processing'].includes(item.status)"
                  class="icon-btn"
                  type="button"
                  title="移除"
                  @click="removeItem(item)"
                >
                  <AppIcon name="x" :size="16" />
                </button>
              </div>
            </li>
          </ul>
        </div>
      </div>

      <!-- 右侧信息栏 -->
      <aside class="upload-side">
        <div class="card side-card">
          <div class="side-card-head">
            <span class="side-icon"><AppIcon name="database" :size="18" /></span>
            <h3>当前知识库</h3>
          </div>
          <template v-if="selectedInfo">
            <p class="side-collection-name">{{ selectedInfo.name }}</p>
            <dl class="mini-stats">
              <div><dt>分块</dt><dd>{{ selectedInfo.count }}</dd></div>
              <div><dt>文件</dt><dd>{{ selectedInfo.files }}</dd></div>
              <div><dt>距离</dt><dd>{{ selectedInfo.distance }}</dd></div>
            </dl>
            <p class="side-note">
              向量模型：{{ selectedInfo.embedding_model || '—' }}
            </p>
          </template>
          <p v-else class="side-note">尚未选择知识库，可先在左侧创建。</p>
        </div>

        <div class="card side-card">
          <div class="side-card-head">
            <span class="side-icon"><AppIcon name="zap" :size="18" /></span>
            <h3>上传时发生了什么</h3>
          </div>
          <ul class="feature-list">
            <li>
              <strong>SHA-256 秒传</strong>
              相同内容再次上传时直接复用结果，不解析、不向量化。
            </li>
            <li>
              <strong>同名覆盖</strong>
              同一文件更新后重新上传，会先删旧分块再写新分块。
            </li>
            <li>
              <strong>向量缓存</strong>
              内容未变的分块从本地缓存读取向量，节省接口费用与推理时间。
            </li>
            <li>
              <strong>并发保护</strong>
              同一文件/内容的并发上传会串行处理，避免重复写入。
            </li>
          </ul>
        </div>

        <div class="card side-card muted-card">
          <div class="side-card-head">
            <span class="side-icon"><AppIcon name="clock" :size="18" /></span>
            <h3>最近上传</h3>
          </div>
          <ul v-if="queue.length" class="recent-list">
            <li v-for="item in queue.slice(-4).reverse()" :key="`r-${item.id}`">
              <span :class="`dot-${item.status}`" />
              <span class="recent-name">{{ item.name }}</span>
              <span class="recent-time">{{ formatTime(item.addedAt) }}</span>
            </li>
          </ul>
          <p v-else class="side-note">本次会话还没有上传记录。</p>
        </div>
      </aside>
    </div>
  </section>
</template>
