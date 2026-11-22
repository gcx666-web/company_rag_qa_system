/**
 * api/client.js —— 后端接口封装。
 *
 * 约定：
 * - 所有请求统一走 `/api` 前缀；开发环境由 Vite 代理到 FastAPI；
 * - 非 2xx 响应统一抛出 ApiError，页面只需展示 error.message；
 * - 上传使用 XMLHttpRequest，以便拿到真实的上传进度（fetch 目前无法读取上传进度）。
 */

const API_BASE = (import.meta.env.VITE_API_BASE || '').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.payload = payload
  }
}

/** 把 FastAPI 的错误响应整理成可读文案（兼容字符串与校验错误数组）。 */
function extractError(payload, status) {
  const detail = payload && payload.detail
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      const loc = Array.isArray(item.loc) ? item.loc.slice(1).join('.') : ''
      return loc ? `${loc}: ${item.msg}` : item.msg
    })
    return parts.join('；') || `请求失败（HTTP ${status}）`
  }
  if (typeof detail === 'string' && detail) return detail
  if (payload && typeof payload.message === 'string') return payload.message
  return `请求失败（HTTP ${status}）`
}

/** 通用 JSON 请求。 */
async function request(path, options = {}) {
  const { method = 'GET', body, headers = {}, signal } = options
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    signal,
    headers: {
      Accept: 'application/json',
      ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })

  const text = await response.text()
  let payload = null
  if (text) {
    try {
      payload = JSON.parse(text)
    } catch {
      payload = { message: text }
    }
  }

  if (!response.ok) {
    throw new ApiError(extractError(payload, response.status), response.status, payload)
  }
  return payload
}

/** 解析 XHR 返回的 JSON 文本。 */
function parseXhrPayload(text) {
  if (!text) return null
  try {
    return JSON.parse(text)
  } catch {
    return { message: text }
  }
}

/**
 * 解析一个 SSE 事件帧（可能包含 event: 与一行或多行 data:）。
 *
 * @param {string} frame 不含结尾空行的帧文本
 * @returns {{event: string, data: any}|null} 没有 data 行时返回 null
 */
function parseSseFrame(frame) {
  let event = 'message'
  const dataLines = []
  for (const line of frame.split(/\r?\n/)) {
    if (!line || line.startsWith(':')) continue
    const index = line.indexOf(':')
    const field = index === -1 ? line : line.slice(0, index)
    let value = index === -1 ? '' : line.slice(index + 1)
    if (value.startsWith(' ')) value = value.slice(1)
    if (field === 'event') event = value
    else if (field === 'data') dataLines.push(value)
  }
  if (!dataLines.length) return null
  const text = dataLines.join('\n')
  let data = text
  try {
    data = JSON.parse(text)
  } catch {
    // 后端始终发送 JSON；解析失败时保留原始文本，便于排查
  }
  return { event, data }
}

/**
 * 流式问答（SSE）。
 *
 * 与普通 fetch 的区别：不等待完整响应体，而是边读 ReadableStream 边解析
 * `event: xxx` / `data: {...}` 帧，逐块回调给页面。
 *
 * @param {Object}   payload             与 /api/chat/ 相同的请求体
 * @param {Object}   options
 * @param {Function} options.onEvent     每帧回调，参数 { event, data }
 * @param {AbortSignal} options.signal   取消信号（清空对话时中断）
 * @returns {Promise<void>} 流结束（done 或 error 事件）后 resolve
 */
async function streamChat(payload, { onEvent, signal } = {}) {
  const response = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    signal,
    headers: {
      Accept: 'text/event-stream',
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const text = await response.text()
    let errorPayload = null
    if (text) {
      try {
        errorPayload = JSON.parse(text)
      } catch {
        errorPayload = { message: text }
      }
    }
    throw new ApiError(extractError(errorPayload, response.status), response.status, errorPayload)
  }
  if (!response.body) {
    throw new ApiError('当前浏览器不支持流式响应', 0)
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''
  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      // SSE 用空行分隔事件帧；服务端固定使用 \n\n
      let boundary = buffer.indexOf('\n\n')
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary)
        buffer = buffer.slice(boundary + 2)
        const parsed = parseSseFrame(frame)
        if (parsed && typeof onEvent === 'function') onEvent(parsed)
        boundary = buffer.indexOf('\n\n')
      }
    }
    // 冲刷解码器剩余字节，并处理最后一帧没有结尾空行的情况
    buffer += decoder.decode()
    const tail = parseSseFrame(buffer)
    if (tail && typeof onEvent === 'function') onEvent(tail)
  } finally {
    // 提前结束（abort）时释放底层流
    try {
      reader.releaseLock()
    } catch {
      /* 已关闭时忽略 */
    }
  }
}

export const api = {
  /** 系统总览：向量模式、缓存、LLM、存储。 */
  systemStatus: () => request('/api/system/status'),

  /** 只取向量模型状态。 */
  embeddingStatus: () => request('/api/system/embedding/status'),

  /** 预热向量模型（离线模式首次会加载本地模型）。 */
  warmupEmbedding: () => request('/api/system/embedding/warmup', { method: 'POST' }),

  /** 清空向量缓存（不删除文档向量）。 */
  clearEmbeddingCache: () =>
    request('/api/system/embedding/cache/clear', { method: 'POST' }),

  /** 知识库集合列表。 */
  listCollections: () => request('/api/document/collections'),

  /** 创建集合。 */
  createCollection: (name) =>
    request('/api/document/collections', { method: 'POST', body: { name } }),

  /** 删除集合。 */
  deleteCollection: (name) =>
    request(`/api/document/collections/${encodeURIComponent(name)}`, {
      method: 'DELETE',
    }),

  /** 集合内文件列表。 */
  listFiles: (collectionName) =>
    request(`/api/document/files?collection_name=${encodeURIComponent(collectionName)}`),

  /** 删除文件。 */
  deleteFile: (collectionName, fileName) =>
    request(
      `/api/document/files?collection_name=${encodeURIComponent(
        collectionName,
      )}&file_name=${encodeURIComponent(fileName)}`,
      { method: 'DELETE' },
    ),

  /** 集合统计。 */
  collectionStats: (collectionName) =>
    request(`/api/document/stats?collection_name=${encodeURIComponent(collectionName)}`),

  /** 支持的文件类型。 */
  supportTypes: () => request('/api/document/support-types'),

  /** RAG 问答（非流式）。 */
  chat: (payload) => request('/api/chat/', { method: 'POST', body: payload }),

  /** RAG 问答（SSE 流式）：逐块回调 onEvent({ event, data })。 */
  chatStream: (payload, options = {}) => streamChat(payload, options),
}

/**
 * 上传单个文件，带进度回调。
 *
 * @param {Object}   options
 * @param {File}     options.file
 * @param {string}   options.collectionName
 * @param {boolean}  options.force          是否忽略秒传，强制重新解析
 * @param {Function} options.onProgress     进度回调，参数 0~1
 * @returns {Promise<Object>} 后端 UploadResponse
 */
export function uploadFile({ file, collectionName, force = false, onProgress }) {
  return new Promise((resolve, reject) => {
    const params = new URLSearchParams()
    if (collectionName) params.set('collection_name', collectionName)
    if (force) params.set('force', 'true')

    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API_BASE}/api/document/upload/?${params.toString()}`)
    xhr.responseType = 'text'
    xhr.timeout = 10 * 60 * 1000 // 大 PDF 解析可能较慢，给 10 分钟

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && typeof onProgress === 'function') {
        onProgress(Math.min(1, event.loaded / Math.max(1, event.total)))
      }
    }

    xhr.onload = () => {
      const payload = parseXhrPayload(xhr.responseText)
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload)
      } else {
        reject(new ApiError(extractError(payload, xhr.status), xhr.status, payload))
      }
    }

    xhr.onerror = () => reject(new ApiError('网络异常，上传失败', 0))
    xhr.ontimeout = () => reject(new ApiError('上传超时，请稍后重试', 0))
    xhr.onabort = () => reject(new ApiError('上传已取消', 0))

    const form = new FormData()
    form.append('file', file, file.name)
    xhr.send(form)
  })
}
