<script setup>
/**
 * SystemView —— 系统状态页面。
 *
 * 展示后端四类优化是否生效，并提供两个运维动作：
 * - 预热向量模型：离线模式首次会加载本地模型，在线模式会建立连接并写缓存；
 * - 清空向量缓存：只清缓存，不影响向量库中的文档。
 *
 * 页面每 30 秒静默刷新一次（仅在页面激活时轮询），缓存命中率会实时变化。
 */
import { computed, onActivated, onDeactivated, onMounted, onUnmounted, ref } from 'vue'
import { api } from '../api/client'
import { notify, refreshSystem, state } from '../store'
import { formatBytes, formatNumber, formatPercent, formatTime, modeLabel } from '../utils/format'
import AppIcon from '../components/AppIcon.vue'

const warming = ref(false)
const clearing = ref(false)
const lastRefreshed = ref(0)
let pollingTimer = null

const embedding = computed(() => state.system?.embedding || {})
const cache = computed(() => embedding.value.cache || {})
const llm = computed(() => state.system?.llm || {})
const features = computed(() => state.system?.features || {})
const storage = computed(() => state.system?.storage || {})
const providers = computed(() => llm.value.providers || [])

const mode = computed(() => embedding.value.mode || '')
const isOffline = computed(() => mode.value === 'offline')
const hitRate = computed(() => {
  const value = Number(cache.value.hit_rate)
  return Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0
})
const ringStyle = computed(() => ({
  background: `conic-gradient(var(--brand) ${hitRate.value * 360}deg, var(--line) 0deg)`,
}))

const cacheEntries = computed(
  () => Number(cache.value.memory_items || 0) + Number(cache.value.disk_entries || 0),
)

const featureCards = computed(() => [
  {
    icon: 'zap',
    title: '向量结果缓存',
    enabled: Boolean(features.value.embedding_cache),
    desc: '内存 LRU + SQLite 持久化，重复文本直接命中，进程重启后仍然有效。',
  },
  {
    icon: 'shield',
    title: '离线向量模式',
    enabled: isOffline.value,
    // 在线模式下离线能力仍然可用，只是当前未启用，显示“可切换”更准确
    available: Boolean(embedding.value.available),
    desc: '本地 SentenceTransformer 推理，设置 HF 离线变量，全程不访问外网。',
  },
  {
    icon: 'database',
    title: '客户端复用',
    enabled: Boolean(features.value.client_reuse),
    desc: 'Chroma / OpenAI / LLM / 解析器按进程（线程）复用，避免每请求重建。',
  },
  {
    icon: 'copy',
    title: '文件秒传',
    enabled: Boolean(features.value.duplicate_detection),
    desc: 'SHA-256 内容指纹 + 上传登记表，相同文件跳过解析与向量化。',
  },
])

onMounted(() => {
  refreshSystem({ silent: true }).then(() => {
    lastRefreshed.value = Date.now()
  })
})

onActivated(() => {
  // KeepAlive 缓存页面：每次切回系统页都刷新一次，并启动 30 秒轮询
  refreshSystem({ silent: true }).then(() => {
    lastRefreshed.value = Date.now()
  })
  stopPolling()
  pollingTimer = window.setInterval(async () => {
    await refreshSystem({ silent: true })
    lastRefreshed.value = Date.now()
  }, 30000)
})

onDeactivated(stopPolling)
onUnmounted(stopPolling)

function stopPolling() {
  if (pollingTimer) {
    window.clearInterval(pollingTimer)
    pollingTimer = null
  }
}

/** 手动刷新。 */
async function refresh() {
  const system = await refreshSystem({ silent: false })
  if (system) lastRefreshed.value = Date.now()
}

/** 预热向量模型。 */
async function warmup() {
  if (warming.value) return
  warming.value = true
  try {
    const result = await api.warmupEmbedding()
    notify(
      `模型预热完成：${result.model}（${result.dimension} 维，${Math.round(
        result.elapsed_ms,
      )} ms）`,
      'success',
      6000,
    )
    await refreshSystem({ silent: true })
    lastRefreshed.value = Date.now()
  } catch (error) {
    notify(error.message, 'error', 7000)
  } finally {
    warming.value = false
  }
}

/** 清空向量缓存。 */
async function clearCache() {
  if (clearing.value) return
  const confirmed = window.confirm(
    '确定清空向量缓存吗？\n只会清空文本→向量的缓存，不会删除向量库中的文档。',
  )
  if (!confirmed) return
  clearing.value = true
  try {
    await api.clearEmbeddingCache()
    notify('向量缓存已清空', 'success')
    await refreshSystem({ silent: true })
    lastRefreshed.value = Date.now()
  } catch (error) {
    notify(error.message, 'error')
  } finally {
    clearing.value = false
  }
}
</script>

<template>
  <section class="view system-view">
    <header class="page-head">
      <div>
        <p class="eyebrow">运行状态 · 性能优化</p>
        <h1>系统状态</h1>
        <p class="page-sub">实时查看向量模型、缓存命中、客户端复用与文件去重配置。</p>
      </div>
      <div class="page-actions">
        <span v-if="lastRefreshed" class="refresh-note">
          更新于 {{ formatTime(lastRefreshed / 1000) }}
        </span>
        <button class="btn btn-ghost" type="button" :disabled="state.systemLoading" @click="refresh">
          <AppIcon name="refresh" :size="16" />
          {{ state.systemLoading ? '刷新中…' : '刷新' }}
        </button>
      </div>
    </header>

    <div v-if="state.systemError" class="inline-alert error">
      <AppIcon name="alert" :size="18" />
      <span>{{ state.systemError }}</span>
    </div>

    <!-- 向量模型主状态 -->
    <div class="card hero-status" :class="{ offline: isOffline }">
      <div class="hero-copy">
        <div class="hero-title-row">
          <span class="mode-pill" :class="isOffline ? 'offline' : 'online'">
            <AppIcon :name="isOffline ? 'shield' : 'cloud'" :size="16" />
            {{ modeLabel(mode) }}
          </span>
          <span class="mode-pill ghost">
            {{ embedding.configured_mode === 'auto' ? '自动选择' : `配置：${embedding.configured_mode}` }}
          </span>
          <span v-if="embedding.loaded" class="mode-pill ghost">模型已加载</span>
        </div>
        <h2 class="hero-model">{{ embedding.model || '尚未配置向量模型' }}</h2>
        <p class="hero-desc">
          <template v-if="isOffline">
            使用本地 SentenceTransformer 推理，HuggingFace 已切换到离线模式；首次加载会读取本机缓存。
          </template>
          <template v-else-if="mode === 'online'">
            通过 OpenAI 兼容接口调用远程向量模型（{{ embedding.base_url || '—' }}）。
          </template>
          <template v-else>
            后端状态不可用，请检查服务日志与 .env 配置。
          </template>
        </p>
        <div class="hero-meta">
          <span><strong>维度</strong>{{ embedding.dimension ?? '加载后获取' }}</span>
          <span><strong>设备</strong>{{ embedding.offline?.device || 'API' }}</span>
          <span><strong>批大小</strong>{{ embedding.batch_size || '—' }}</span>
          <span><strong>可用</strong>{{ embedding.available ? '是' : '否' }}</span>
        </div>
        <p v-if="embedding.error" class="hero-error">
          <AppIcon name="alert" :size="15" />
          {{ embedding.error }}
        </p>
      </div>

      <div class="hero-actions">
        <button
          class="btn btn-primary"
          type="button"
          :disabled="warming || !embedding.available"
          @click="warmup"
        >
          <AppIcon name="zap" :size="16" />
          {{ warming ? '预热中…' : '预热向量模型' }}
        </button>
        <button
          class="btn btn-ghost"
          type="button"
          :disabled="clearing || !cache.enabled"
          @click="clearCache"
        >
          <AppIcon name="trash" :size="16" />
          {{ clearing ? '清理中…' : '清空向量缓存' }}
        </button>
      </div>
    </div>

    <div class="system-grid">
      <!-- 缓存卡片 -->
      <div class="card cache-card">
        <header class="card-head compact">
          <div>
            <h3>向量缓存命中</h3>
            <p>{{ cache.enabled ? '内存 + 磁盘双层缓存' : '缓存已关闭' }}</p>
          </div>
          <span class="tag" :class="cache.enabled ? 'tag-success' : 'tag-muted'">
            {{ cache.enabled ? '已启用' : '未启用' }}
          </span>
        </header>

        <div class="cache-body">
          <div class="hit-ring-wrap">
            <div class="hit-ring" :style="ringStyle">
              <div class="hit-ring-inner">
                <strong>{{ formatPercent(hitRate) }}</strong>
                <span>命中率</span>
              </div>
            </div>
          </div>

          <dl class="cache-stats">
            <div><dt>命中</dt><dd>{{ formatNumber(cache.hits) }}</dd></div>
            <div><dt>未命中</dt><dd>{{ formatNumber(cache.misses) }}</dd></div>
            <div><dt>内存条目</dt><dd>{{ formatNumber(cache.memory_items) }}</dd></div>
            <div><dt>磁盘条目</dt><dd>{{ formatNumber(cache.disk_entries) }}</dd></div>
            <div><dt>磁盘占用</dt><dd>{{ formatBytes(cache.disk_bytes) }}</dd></div>
            <div><dt>写缓存次数</dt><dd>{{ formatNumber(cache.writes) }}</dd></div>
          </dl>
        </div>

        <p class="cache-path" :title="cache.db_path">
          <AppIcon name="hard-drive" :size="14" />
          {{ cache.db_path || '仅内存缓存' }}
        </p>
      </div>

      <!-- 优化项卡片 -->
      <div class="card features-card">
        <header class="card-head compact">
          <div>
            <h3>后端优化项</h3>
            <p>本次升级的四项核心改动</p>
          </div>
        </header>
        <ul class="optimization-list">
          <li v-for="item in featureCards" :key="item.title">
            <span class="optimization-icon" :class="{ off: !item.enabled }">
              <AppIcon :name="item.icon" :size="18" />
            </span>
            <div>
              <p class="optimization-title">
                {{ item.title }}
                <span
                  class="tag"
                  :class="
                    item.enabled
                      ? 'tag-success'
                      : item.available
                        ? 'tag-brand'
                        : 'tag-muted'
                  "
                >
                  {{ item.enabled ? '生效中' : item.available ? '可切换' : '未启用' }}
                </span>
              </p>
              <p class="optimization-desc">{{ item.desc }}</p>
            </div>
          </li>
        </ul>
      </div>

      <!-- LLM 供应商 -->
      <div class="card llm-card">
        <header class="card-head compact">
          <div>
            <h3>大模型供应商</h3>
            <p>默认：{{ llm.default_provider || '未配置' }}</p>
          </div>
          <span class="tag tag-soft">温度 {{ llm.temperature }} · 上限 {{ llm.max_tokens }}</span>
        </header>
        <div class="provider-list">
          <div
            v-for="item in providers"
            :key="item.name"
            class="provider-row"
            :class="{ disabled: !item.configured }"
          >
            <span class="provider-dot" :class="item.configured ? 'ok' : 'off'" />
            <div class="provider-copy">
              <strong>{{ item.name }}</strong>
              <small>{{ item.model || '未配置模型' }}</small>
            </div>
            <span v-if="item.is_default" class="tag tag-brand">默认</span>
            <span class="tag" :class="item.configured ? 'tag-success' : 'tag-muted'">
              {{ item.configured ? '已配置' : '缺少密钥' }}
            </span>
          </div>
        </div>
      </div>

      <!-- 存储与运行环境 -->
      <div class="card storage-card">
        <header class="card-head compact">
          <div>
            <h3>存储与运行环境</h3>
            <p>秒传登记表 {{ formatNumber(storage.registry_records) }} 条记录</p>
          </div>
        </header>
        <dl class="path-list">
          <div>
            <dt>向量库目录</dt>
            <dd :title="storage.chroma_dir">{{ storage.chroma_dir }}</dd>
          </div>
          <div>
            <dt>上传目录</dt>
            <dd :title="storage.upload_dir">{{ storage.upload_dir }}</dd>
          </div>
          <div>
            <dt>上传登记表</dt>
            <dd :title="storage.registry_db">{{ storage.registry_db }}</dd>
          </div>
          <div>
            <dt>缓存目录</dt>
            <dd :title="storage.cache_dir">{{ storage.cache_dir }}</dd>
          </div>
          <div>
            <dt>知识库 / 登记记录</dt>
            <dd>{{ storage.collections }} / {{ storage.registry_records }}</dd>
          </div>
          <div>
            <dt>文件去重</dt>
            <dd>{{ storage.dedup_enabled ? '开启' : '关闭' }}</dd>
          </div>
        </dl>
        <p class="env-note">
          离线相关环境变量：
          <code>HF_HUB_OFFLINE={{ embedding.offline?.env?.HF_HUB_OFFLINE || '未设置' }}</code>
          <code>TRANSFORMERS_OFFLINE={{ embedding.offline?.env?.TRANSFORMERS_OFFLINE || '未设置' }}</code>
        </p>
      </div>
    </div>
  </section>
</template>
