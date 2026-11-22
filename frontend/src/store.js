/**
 * store.js —— 轻量全局状态。
 *
 * 项目只有四个页面、状态也不复杂，因此不引入 Pinia，直接用 Vue 的 reactive
 * 对象做一个共享 store；需要跨页面共享的只有系统状态、知识库列表和 Toast。
 */

import { reactive } from 'vue'
import { api } from './api/client'

export const state = reactive({
  // 系统总览（向量模式、缓存命中率、LLM 供应商、存储目录）
  system: null,
  systemLoading: false,
  systemError: '',
  // 知识库集合列表
  collections: [],
  collectionsLoading: false,
  collectionsError: '',
  // 全局 Toast
  toasts: [],
  // 上传页支持的文件后缀（由后端返回，失败时用兜底值）
  supportTypes: ['.pdf', '.docx', '.md', '.html', '.htm'],
  supportTypesLoaded: false,
})

let toastSeed = 0

/** 主动关闭某条 Toast。 */
export function dismissToast(id) {
  const index = state.toasts.findIndex((item) => item.id === id)
  if (index >= 0) state.toasts.splice(index, 1)
}

/**
 * 弹出一条 Toast。
 * @param {string} message
 * @param {'success'|'error'|'warning'|'info'} type
 * @param {number} duration 自动消失时间（毫秒）；0 表示不自动消失
 */
export function notify(message, type = 'info', duration = 4200) {
  toastSeed += 1
  const id = toastSeed
  state.toasts.push({ id, message: String(message || ''), type })
  // 最多保留 4 条，避免刷屏
  if (state.toasts.length > 4) state.toasts.shift()
  if (duration > 0) {
    window.setTimeout(() => dismissToast(id), duration)
  }
  return id
}

/** 刷新系统总览；silent=true 时失败不弹 Toast（用于后台轮询）。 */
export async function refreshSystem({ silent = false } = {}) {
  state.systemLoading = true
  try {
    state.system = await api.systemStatus()
    state.systemError = ''
    // 文件类型单独从上传接口获取（系统总览不重复返回）
    if (!state.supportTypesLoaded) {
      try {
        const payload = await api.supportTypes()
        state.supportTypes = payload?.support_file_types || state.supportTypes
        state.supportTypesLoaded = true
      } catch {
        // 获取失败保持兜底列表，不影响页面
      }
    }
    return state.system
  } catch (error) {
    state.systemError = error.message || '无法连接后端服务'
    if (!silent) notify(state.systemError, 'error')
    return null
  } finally {
    state.systemLoading = false
  }
}

/** 刷新知识库集合列表。 */
export async function refreshCollections({ silent = false } = {}) {
  state.collectionsLoading = true
  try {
    const payload = await api.listCollections()
    state.collections = payload?.collections || []
    state.collectionsError = ''
    return state.collections
  } catch (error) {
    state.collectionsError = error.message || '知识库列表加载失败'
    if (!silent) notify(state.collectionsError, 'error')
    return null
  } finally {
    state.collectionsLoading = false
  }
}

/** 取默认集合：优先 default，其次第一个。 */
export function defaultCollectionName() {
  if (!state.collections.length) return ''
  const preferred = state.collections.find((item) => item.name === 'default')
  return (preferred || state.collections[0]).name
}
