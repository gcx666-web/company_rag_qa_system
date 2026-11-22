<script setup>
/**
 * AppIcon —— 统一的内联 SVG 图标集。
 * 不引入图标库，减少依赖与打包体积；图标使用 currentColor，跟随文字颜色。
 */
defineProps({
  name: { type: String, required: true },
  size: { type: [Number, String], default: 20 },
})
</script>

<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="1.8"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <template v-if="name === 'chat'">
      <path d="M21 11.5a8.4 8.4 0 0 1-8.5 8.5 8.7 8.7 0 0 1-3.9-.9L3 21l1.9-5.6a8.4 8.4 0 0 1-.9-3.9A8.5 8.5 0 0 1 12.5 3 8.4 8.4 0 0 1 21 11.5Z" />
    </template>
    <template v-else-if="name === 'upload'">
      <path d="M12 16V4m0 0L7 9m5-5 5 5" />
      <path d="M4 16v2a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-2" />
    </template>
    <template v-else-if="name === 'database'">
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
      <path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
    </template>
    <template v-else-if="name === 'activity'">
      <path d="M3 12h4l2.5-7 5 14 2.5-7H21" />
    </template>
    <template v-else-if="name === 'refresh'">
      <path d="M20 11a8 8 0 1 0-2.3 6.3" />
      <path d="M20 4v7h-7" />
    </template>
    <template v-else-if="name === 'trash'">
      <path d="M4 7h16M9 7V4h6v3m-8 0 1 14h8l1-14" />
      <path d="M10 11v6m4-6v6" />
    </template>
    <template v-else-if="name === 'plus'">
      <path d="M12 5v14M5 12h14" />
    </template>
    <template v-else-if="name === 'send'">
      <path d="m22 2-7 20-4-9-9-4 20-7Z" />
      <path d="M22 2 11 13" />
    </template>
    <template v-else-if="name === 'file'">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z" />
      <path d="M14 3v5h5" />
    </template>
    <template v-else-if="name === 'spark'">
      <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" />
      <path d="M19 16.5 20 19l2.5 1-2.5 1-1 2.5" />
    </template>
    <template v-else-if="name === 'check'">
      <path d="m5 12 4.5 4.5L19 7" />
    </template>
    <template v-else-if="name === 'alert'">
      <path d="M12 3 2.5 20h19L12 3Z" />
      <path d="M12 9v5m0 3v.5" />
    </template>
    <template v-else-if="name === 'search'">
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.5-3.5" />
    </template>
    <template v-else-if="name === 'cpu'">
      <rect x="6" y="6" width="12" height="12" rx="2" />
      <path d="M9 2v2m6-2v2M9 20v2m6-2v2M2 9h2m-2 6h2m16-6h2m-2 6h2" />
      <rect x="10" y="10" width="4" height="4" rx="1" />
    </template>
    <template v-else-if="name === 'hard-drive'">
      <path d="M21 12H3m18 0-3.5-7H6.5L3 12m18 0v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-6" />
      <path d="M7 16h.5m4.5 0h.5" />
    </template>
    <template v-else-if="name === 'cloud'">
      <path d="M7 18a4.5 4.5 0 0 1-.5-9 6 6 0 0 1 11.6 1.4A3.8 3.8 0 0 1 17.5 18Z" />
    </template>
    <template v-else-if="name === 'shield'">
      <path d="M12 3 5 6v6c0 4.5 3 7.5 7 9 4-1.5 7-4.5 7-9V6l-7-3Z" />
      <path d="m9 12 2 2 4-4" />
    </template>
    <template v-else-if="name === 'clock'">
      <circle cx="12" cy="12" r="9" />
      <path d="M12 7v5l3 2" />
    </template>
    <template v-else-if="name === 'x'">
      <path d="m6 6 12 12M18 6 6 18" />
    </template>
    <template v-else-if="name === 'layers'">
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 13 9 5 9-5M3 18l9 5 9-5" />
    </template>
    <template v-else-if="name === 'zap'">
      <path d="M13 2 4 14h7l-1 8 9-12h-7l1-8Z" />
    </template>
    <template v-else-if="name === 'copy'">
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15V5a2 2 0 0 1 2-2h10" />
    </template>
    <template v-else>
      <circle cx="12" cy="12" r="9" />
      <path d="M12 8v5m0 3v.5" />
    </template>
  </svg>
</template>
