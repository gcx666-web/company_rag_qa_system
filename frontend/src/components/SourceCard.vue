<script setup>
/** 引用来源卡片：文件名、页码、块类型、相似度与原文片段。 */
import { computed } from 'vue'
import { scorePercent, scoreTone } from '../utils/format'
import AppIcon from './AppIcon.vue'

const props = defineProps({
  source: { type: Object, required: true },
})

const percent = computed(() => scorePercent(props.source.score))
const tone = computed(() => scoreTone(props.source.score))
</script>

<template>
  <article class="source-card">
    <header class="source-head">
      <div class="source-title">
        <span class="source-file-icon">
          <AppIcon name="file" :size="15" />
        </span>
        <strong>{{ source.file_name || '未知文件' }}</strong>
        <span v-if="source.page" class="tag">第 {{ source.page }} 页</span>
        <span class="tag tag-soft">
          {{ source.chunk_type === 'table' ? '表格' : '正文' }}
        </span>
      </div>
      <span class="score-pill" :class="`score-${tone}`" :title="`相似度 ${percent}`">
        {{ percent }}
      </span>
    </header>
    <p class="source-content">{{ source.content }}</p>
  </article>
</template>
