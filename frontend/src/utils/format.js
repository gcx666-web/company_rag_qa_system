/** utils/format.js —— 展示层格式化工具。 */

/** 字节数转可读大小。 */
export function formatBytes(bytes) {
  const value = Number(bytes)
  if (!Number.isFinite(value) || value <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)))
  const size = value / 1024 ** index
  return `${size >= 100 || index === 0 ? Math.round(size) : size.toFixed(1)} ${units[index]}`
}

/** 数字千分位。 */
export function formatNumber(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '0'
  return number.toLocaleString('zh-CN')
}

/** 0~1 的比例转百分比文案。 */
export function formatPercent(value, digits = 1) {
  const number = Number(value)
  if (!Number.isFinite(number)) return '0%'
  return `${(number * 100).toFixed(digits)}%`
}

/** 秒级时间戳转本地时间。 */
export function formatTime(seconds) {
  const value = Number(seconds)
  if (!Number.isFinite(value) || value <= 0) return '—'
  const date = new Date(value * 1000)
  const pad = (part) => String(part).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`
}

/** 秒级时间戳转“x 分钟前”。 */
export function timeAgo(seconds) {
  const value = Number(seconds)
  if (!Number.isFinite(value) || value <= 0) return '—'
  const diff = Math.max(0, Date.now() / 1000 - value)
  if (diff < 60) return '刚刚'
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`
  if (diff < 86400 * 30) return `${Math.floor(diff / 86400)} 天前`
  return formatTime(value)
}

/** 相似度分数转百分比（Chroma cosine 场景 score 在 0~1）。 */
export function scorePercent(score) {
  const value = Number(score)
  if (!Number.isFinite(value)) return '0%'
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`
}

/** 向量模式文案。 */
export function modeLabel(mode) {
  if (mode === 'offline') return '离线模式'
  if (mode === 'online') return '在线模式'
  return '未就绪'
}

/** 按相似度返回颜色类名。 */
export function scoreTone(score) {
  const value = Number(score)
  if (value >= 0.75) return 'high'
  if (value >= 0.5) return 'mid'
  return 'low'
}
