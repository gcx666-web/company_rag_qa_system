<script setup>
/** 左侧导航：四个功能入口 + 底部向量模式状态。 */
import { computed } from 'vue'
import { state } from '../store'
import { modeLabel } from '../utils/format'
import AppIcon from './AppIcon.vue'

const props = defineProps({
  current: { type: String, default: 'chat' },
})
defineEmits(['navigate'])

const navItems = [
  { key: 'chat', label: '对话问答', hint: '基于知识库的 RAG 回答', icon: 'chat' },
  { key: 'upload', label: '文档上传', hint: '解析、去重与向量化', icon: 'upload' },
  { key: 'knowledge', label: '知识库', hint: '集合与文件管理', icon: 'database' },
  { key: 'system', label: '系统状态', hint: '模型、缓存与运行信息', icon: 'activity' },
]

const embedding = computed(() => state.system?.embedding || null)
const knowledgeCount = computed(() => state.collections.length)

const statusText = computed(() => {
  if (state.systemError) return '后端未连接'
  if (!embedding.value) return '正在读取状态…'
  if (embedding.value.error) return '向量模型配置异常'
  return `${modeLabel(embedding.value.mode)} · ${embedding.value.model || '未配置模型'}`
})

const statusTone = computed(() => {
  if (state.systemError || embedding.value?.error) return 'danger'
  if (!embedding.value) return 'muted'
  return embedding.value.mode === 'offline' ? 'offline' : 'online'
})
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">
        <AppIcon name="layers" :size="22" />
      </div>
      <div class="brand-copy">
        <strong>企业知识库</strong>
        <span>RAG 智能问答系统</span>
      </div>
    </div>

    <nav class="side-nav" aria-label="主导航">
      <button
        v-for="item in navItems"
        :key="item.key"
        type="button"
        class="side-nav-item"
        :class="{ active: props.current === item.key }"
        @click="$emit('navigate', item.key)"
      >
        <span class="side-nav-icon">
          <AppIcon :name="item.icon" :size="19" />
        </span>
        <span class="side-nav-copy">
          <span class="side-nav-label">
            {{ item.label }}
            <em v-if="item.key === 'knowledge' && knowledgeCount" class="nav-badge">
              {{ knowledgeCount }}
            </em>
          </span>
          <span class="side-nav-hint">{{ item.hint }}</span>
        </span>
      </button>
    </nav>

    <div class="sidebar-spacer" />

    <button
      type="button"
      class="side-status"
      :class="`tone-${statusTone}`"
      @click="$emit('navigate', 'system')"
      title="查看向量模型与缓存详情"
    >
      <span class="status-dot" />
      <span class="side-status-copy">
        <span class="side-status-title">向量模型</span>
        <span class="side-status-value">{{ statusText }}</span>
      </span>
    </button>

    <p class="sidebar-foot">向量缓存 · 文件秒传 · 离线可用</p>
  </aside>
</template>
