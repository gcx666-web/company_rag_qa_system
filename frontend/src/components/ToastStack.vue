<script setup>
/** 全局 Toast 容器：右下角浮层，按类型着色，可手动关闭。 */
import { state, dismissToast } from '../store'
import AppIcon from './AppIcon.vue'

const iconByType = {
  success: 'check',
  error: 'alert',
  warning: 'alert',
  info: 'spark',
}
</script>

<template>
  <div class="toast-stack" aria-live="polite">
    <TransitionGroup name="toast">
      <div
        v-for="toast in state.toasts"
        :key="toast.id"
        class="toast"
        :class="`toast-${toast.type}`"
      >
        <span class="toast-icon">
          <AppIcon :name="iconByType[toast.type] || 'spark'" :size="18" />
        </span>
        <p class="toast-message">{{ toast.message }}</p>
        <button class="toast-close" type="button" @click="dismissToast(toast.id)">
          <AppIcon name="x" :size="15" />
        </button>
      </div>
    </TransitionGroup>
  </div>
</template>
