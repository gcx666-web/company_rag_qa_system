<script setup>
/**
 * App.vue —— 应用外壳。
 *
 * 四个页面用 KeepAlive 缓存：切回对话页时保留对话记录，切回上传页时保留队列。
 * 首次加载时并行拉取系统状态与知识库列表，供各页面共享。
 */
import { computed, onMounted, ref } from 'vue'
import SidebarNav from './components/SidebarNav.vue'
import ToastStack from './components/ToastStack.vue'
import ChatView from './views/ChatView.vue'
import UploadView from './views/UploadView.vue'
import KnowledgeView from './views/KnowledgeView.vue'
import SystemView from './views/SystemView.vue'
import { refreshCollections, refreshSystem } from './store'

const view = ref('chat')
const views = {
  chat: ChatView,
  upload: UploadView,
  knowledge: KnowledgeView,
  system: SystemView,
}

const currentComponent = computed(() => views[view.value] || ChatView)

function navigate(next) {
  if (views[next]) view.value = next
}

onMounted(async () => {
  // 并行请求，任何一个失败都由 store 自己提示，不阻塞页面渲染
  await Promise.all([
    refreshSystem({ silent: true }),
    refreshCollections({ silent: true }),
  ])
})
</script>

<template>
  <div class="app-shell">
    <SidebarNav :current="view" @navigate="navigate" />

    <main class="app-main">
      <div class="app-main-inner">
        <KeepAlive>
          <component :is="currentComponent" />
        </KeepAlive>
      </div>
    </main>

    <ToastStack />
  </div>
</template>
