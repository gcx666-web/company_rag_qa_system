import { createApp } from 'vue'
import App from './App.vue'
import './styles/main.css'

// 应用入口：单页应用，不引入 vue-router，页面切换由 App.vue 的 nav 状态控制，
// 这样打包体积更小，也避免为四个页面维护路由表。
createApp(App).mount('#app')
