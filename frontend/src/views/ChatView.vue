<script setup>
/**
 * ChatView —— RAG 对话页面。
 *
 * 交互要点：
 * - 选择知识库与大模型供应商后提问；
 * - 通过 `/api/chat/stream`（SSE）逐块接收答案，边生成边渲染；
 * - 回答用 Markdown 渲染（经过 DOMPurify 清洗）；
 * - 每条回答下方展示命中的原文片段（来源卡片），支持溯源；
 * - Enter 发送、Shift+Enter 换行，发送中禁用输入避免重复提交。
 */
import { computed, nextTick, onMounted, ref, watch } from 'vue'
import { api } from '../api/client'
import { notify, refreshCollections, refreshSystem, state } from '../store'
import { renderMarkdown } from '../utils/markdown'
import AppIcon from '../components/AppIcon.vue'
import SourceCard from '../components/SourceCard.vue'

const messages = ref([])
const question = ref('')
const sending = ref(false)
const collection = ref('')
const provider = ref('')
const topK = ref(5)
const threadEl = ref(null)
const textareaEl = ref(null)
let messageSeed = 0
let requestSeed = 0
let activeController = null

const suggestions = [
  '请概括一下知识库里的主要内容',
  '公司有哪些核心业务？',
  '请列出文档中提到的关键流程',
]

const usableProviders = computed(() =>
  (state.system?.llm?.providers || []).filter((item) => item.configured),
)

const embeddingMode = computed(() => state.system?.embedding?.mode || '')

const selectedInfo = computed(() =>
  state.collections.find((item) => item.name === collection.value),
)

// 旧知识库可能是在另一个向量模型下建立的，维度不一致时 Chroma 会拒绝检索；
// 这里提前提示并禁用发送，避免用户提交后才看到 409。
const incompatible = computed(() => selectedInfo.value?.compatible === false)

// 集合列表加载完成后自动选中默认集合
watch(
  () => state.collections,
  (collections) => {
    if (!collections.length) {
      if (collection.value) collection.value = ''
      return
    }
    const exists = collections.some((item) => item.name === collection.value)
    if (!exists) {
      collection.value =
        collections.find((item) => item.name === 'default')?.name || collections[0].name
    }
  },
  { immediate: true, deep: true },
)

// 系统状态加载完成后自动选中默认供应商
watch(
  usableProviders,
  (providers) => {
    if (!providers.length) return
    const exists = providers.some((item) => item.name === provider.value)
    if (!exists) {
      provider.value = (providers.find((item) => item.is_default) || providers[0]).name
    }
  },
  { immediate: true },
)

onMounted(() => {
  if (!state.collections.length) refreshCollections({ silent: true })
  if (!state.system) refreshSystem({ silent: true })
  nextTick(growTextarea)
})

/** 滚动到底部（新消息 / 回答完成时调用）。 */
function scrollToBottom() {
  nextTick(() => {
    const el = threadEl.value
    if (el) el.scrollTop = el.scrollHeight
  })
}

/** 输入框跟随内容自动增高，最高 180px。 */
function growTextarea() {
  const el = textareaEl.value
  if (!el) return
  el.style.height = 'auto'
  el.style.height = `${Math.min(el.scrollHeight, 180)}px`
}

/** 发送问题（SSE 流式渲染，逐块追加答案）。 */
async function send(preset) {
  const text = String(preset ?? question.value).trim()
  if (!text || sending.value) return
  if (!collection.value) {
    notify('请先创建并选择知识库', 'warning')
    return
  }
  if (incompatible.value) {
    notify(
      '当前知识库的向量维度与向量模型不一致，请新建知识库重新导入文档',
      'warning',
      7000,
    )
    return
  }
  if (!provider.value) {
    notify('请先在 .env 配置至少一个大模型供应商', 'warning')
    return
  }

  messageSeed += 1
  messages.value.push({
    id: `u-${messageSeed}`,
    role: 'user',
    content: text,
  })

  messages.value.push({
    id: `a-${messageSeed}`,
    role: 'assistant',
    content: '',
    sources: [],
    pending: true,
    error: '',
    meta: null,
  })
  // 取数组中的响应式代理：直接改 push 进去的原对象不会触发视图刷新
  const assistant = messages.value[messages.value.length - 1]

  question.value = ''
  nextTick(growTextarea)
  scrollToBottom()

  sending.value = true
  requestSeed += 1
  const requestId = requestSeed
  activeController = new AbortController()
  let sawDone = false
  try {
    await api.chatStream(
      {
        question: text,
        model: provider.value,
        collection_name: collection.value,
        top_k: Number(topK.value),
      },
      {
        signal: activeController.signal,
        onEvent: ({ event, data }) => {
          // 只更新本次请求对应的消息，避免用户中途清空对话造成错位
          if (requestId !== requestSeed) return
          if (event === 'meta') {
            // 来源卡片与检索耗时在流开始时就返回，先展示出来
            assistant.sources = Array.isArray(data?.sources) ? data.sources : []
            assistant.meta = {
              model: data?.model || '',
              totalTime: 0,
              retrievalTime: data?.retrieval_time_ms || 0,
              tokens: {},
            }
          } else if (event === 'delta') {
            // 首个增量到达后收起“正在生成”动画，开始逐字输出
            assistant.pending = false
            assistant.content += data?.content || ''
            scrollToBottom()
          } else if (event === 'done') {
            sawDone = true
            assistant.pending = false
            assistant.content = data?.answer || assistant.content
            assistant.meta = {
              model: data?.model || assistant.meta?.model || '',
              totalTime: data?.total_time_ms || 0,
              retrievalTime:
                data?.retrieval_time_ms ?? assistant.meta?.retrievalTime ?? 0,
              tokens: data?.tokens_used || {},
            }
            scrollToBottom()
          } else if (event === 'error') {
            assistant.pending = false
            assistant.error = data?.message || '提问失败'
            notify(assistant.error, 'error')
          }
        },
      },
    )
    if (requestId !== requestSeed) return
    if (!assistant.error && !sawDone) {
      assistant.error = '回答生成中断，请重试'
      notify(assistant.error, 'error')
    } else if (!assistant.error && sawDone && !assistant.content) {
      assistant.error = '模型没有返回内容，请重试或更换模型'
      notify(assistant.error, 'error')
    }
  } catch (error) {
    if (requestId !== requestSeed) return
    assistant.pending = false
    if (error?.name === 'AbortError') return
    assistant.error = error.message || '提问失败'
    notify(assistant.error, 'error')
  } finally {
    if (requestId === requestSeed) {
      sending.value = false
      activeController = null
    }
    assistant.pending = false
    scrollToBottom()
  }
}

/** 点击示例问题：填入并直接发送。 */
function useSuggestion(text) {
  question.value = text
  nextTick(growTextarea)
  send(text)
}

/** 清空当前对话。 */
function clearChat() {
  if (!messages.value.length) return
  if (!window.confirm('确定清空当前对话吗？')) return
  // 中断仍在进行的流式请求，避免后台继续生成
  activeController?.abort()
  activeController = null
  requestSeed += 1
  messages.value = []
  sending.value = false
}

/** token 用量文案。 */
function tokensText(meta) {
  const tokens = meta?.tokens || {}
  const total = Number(tokens.total_tokens ?? tokens.total ?? 0)
  return total > 0 ? `${total} tokens` : '—'
}
</script>

<template>
  <section class="view chat-view">
    <header class="page-head">
      <div>
        <p class="eyebrow">RAG · 知识增强问答</p>
        <h1>对话问答</h1>
        <p class="page-sub">检索知识库中的相关片段，交给大模型生成有依据的回答。</p>
      </div>
      <div class="page-actions">
        <button class="btn btn-ghost" type="button" :disabled="!messages.length" @click="clearChat">
          <AppIcon name="trash" :size="16" />
          清空对话
        </button>
      </div>
    </header>

    <div class="chat-layout">
      <div class="chat-toolbar">
        <label class="field">
          <span class="field-label">知识库</span>
          <select v-model="collection" :disabled="sending" class="input select">
            <option value="" disabled>请选择知识库</option>
            <option v-for="item in state.collections" :key="item.name" :value="item.name">
              {{ item.name }}（{{ item.count }} 个分块）
            </option>
          </select>
        </label>

        <label class="field">
          <span class="field-label">大模型</span>
          <select v-model="provider" :disabled="sending" class="input select">
            <option value="" disabled>请选择供应商</option>
            <option v-for="item in usableProviders" :key="item.name" :value="item.name">
              {{ item.name }} · {{ item.model || '未命名模型' }}
            </option>
          </select>
        </label>

        <label class="field field-compact">
          <span class="field-label">召回数量</span>
          <select v-model="topK" :disabled="sending" class="input select">
            <option :value="3">3 条</option>
            <option :value="5">5 条</option>
            <option :value="8">8 条</option>
            <option :value="10">10 条</option>
          </select>
        </label>

        <span class="toolbar-spacer" />
        <span class="mode-chip" :class="embeddingMode === 'offline' ? 'offline' : 'online'">
          <AppIcon :name="embeddingMode === 'offline' ? 'shield' : 'cloud'" :size="14" />
          {{ embeddingMode === 'offline' ? '离线向量' : '在线向量' }}
        </span>
      </div>

      <div v-if="incompatible" class="inline-alert error compat-banner">
        <AppIcon name="alert" :size="18" />
        <span>
          知识库「{{ selectedInfo.name }}」的向量维度为
          {{ selectedInfo.embedding_dimension }}，当前模型输出
          {{ selectedInfo.model_dimension }} 维，无法检索。请新建知识库重新导入文档，
          或切回建立该知识库时使用的向量模型。
        </span>
      </div>

      <div ref="threadEl" class="chat-thread">
        <!-- 空状态：引导用户提问 -->
        <div v-if="!messages.length" class="chat-empty">
          <div class="empty-orb">
            <AppIcon name="spark" :size="30" />
          </div>
          <h2>向你的知识库提问</h2>
          <p>
            系统会先在向量库中检索最相关的{{ topK }} 个片段，再把它们作为上下文交给大模型，
            回答末尾可以查看引用来源。
          </p>

          <div v-if="!state.collections.length" class="inline-alert">
            <AppIcon name="alert" :size="18" />
            <span>还没有知识库，请先到「文档上传」创建集合并导入文档。</span>
          </div>
          <div v-else-if="!usableProviders.length" class="inline-alert">
            <AppIcon name="alert" :size="18" />
            <span>尚未配置可用的大模型密钥，请检查后端 .env 配置。</span>
          </div>

          <div class="suggestion-list">
            <button
              v-for="item in suggestions"
              :key="item"
              type="button"
              class="suggestion"
              @click="useSuggestion(item)"
            >
              <AppIcon name="search" :size="15" />
              {{ item }}
            </button>
          </div>
        </div>

        <!-- 消息列表 -->
        <article
          v-for="message in messages"
          :key="message.id"
          class="message"
          :class="`message-${message.role}`"
        >
          <div class="message-avatar">
            <AppIcon :name="message.role === 'user' ? 'chat' : 'spark'" :size="18" />
          </div>

          <div class="message-body">
            <div v-if="message.role === 'user'" class="message-user-text">
              {{ message.content }}
            </div>

            <template v-else>
              <div v-if="message.pending" class="typing" aria-label="正在生成回答">
                <span /><span /><span />
              </div>

              <div v-if="message.error" class="message-error">
                <AppIcon name="alert" :size="17" />
                <span>{{ message.error }}</span>
              </div>

              <!-- 流式增量可能先到一段内容，出错时保留已经生成的部分 -->
              <div
                v-if="message.content"
                class="md-body"
                v-html="renderMarkdown(message.content)"
              />

              <details
                v-if="message.sources && message.sources.length"
                class="source-group"
                :open="message.sources.length <= 3"
              >
                <summary>
                  <span class="source-summary">
                    <AppIcon name="file" :size="15" />
                    引用来源
                    <em>{{ message.sources.length }}</em>
                  </span>
                  <span class="source-summary-hint">点击展开 / 收起</span>
                </summary>
                <div class="source-list">
                  <SourceCard
                    v-for="(source, index) in message.sources"
                    :key="source.id || index"
                    :source="source"
                  />
                </div>
              </details>

              <footer v-if="message.meta" class="message-meta">
                <span v-if="message.meta.model">
                  <AppIcon name="cpu" :size="13" />
                  {{ message.meta.model }}
                </span>
                <span>
                  <AppIcon name="clock" :size="13" />
                  总耗时 {{ Math.round(message.meta.totalTime) }} ms
                </span>
                <span v-if="message.meta.retrievalTime">
                  检索 {{ Math.round(message.meta.retrievalTime) }} ms
                </span>
                <span>{{ tokensText(message.meta) }}</span>
              </footer>
            </template>
          </div>
        </article>
      </div>

      <form class="composer" @submit.prevent="send()">
        <textarea
          ref="textareaEl"
          v-model="question"
          class="composer-input"
          rows="1"
          placeholder="输入问题，Enter 发送，Shift + Enter 换行…"
          :disabled="sending"
          @input="growTextarea"
          @keydown.enter.exact.prevent="send()"
        />
        <div class="composer-footer">
          <span class="composer-hint">
            <AppIcon name="shield" :size="14" />
            回答由大模型生成，请结合引用来源核对
          </span>
          <button
            class="btn btn-primary composer-send"
            type="submit"
            :disabled="sending || !question.trim() || incompatible"
          >
            <template v-if="sending">生成中…</template>
            <template v-else>
              发送
              <AppIcon name="send" :size="16" />
            </template>
          </button>
        </div>
      </form>
    </div>
  </section>
</template>
