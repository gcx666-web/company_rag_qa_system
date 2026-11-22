/**
 * utils/markdown.js —— 把大模型返回的 Markdown 安全地渲染成 HTML。
 *
 * 大模型输出属于“不可信内容”，必须经过 DOMPurify 清洗后再 v-html，
 * 避免模型被提示词注入后输出 <script> 或 onerror 之类的 XSS 载荷。
 */

import { marked } from 'marked'
import DOMPurify from 'dompurify'

marked.setOptions({
  gfm: true, // 支持表格、任务列表、删除线
  breaks: true, // 单个换行也渲染成 <br>，更符合问答阅读习惯
})

// 让正文里的链接在新标签页打开，并阻断 opener 引用
DOMPurify.addHook('afterSanitizeAttributes', (node) => {
  if (node.tagName === 'A') {
    node.setAttribute('target', '_blank')
    node.setAttribute('rel', 'noopener noreferrer')
  }
})

/**
 * 渲染 Markdown 为安全 HTML。
 * @param {string} text
 * @returns {string}
 */
export function renderMarkdown(text) {
  const source = String(text ?? '')
  if (!source.trim()) return ''
  try {
    return DOMPurify.sanitize(marked.parse(source))
  } catch {
    // 解析失败时退化为纯文本（仍然转义），保证页面不崩
    const escaped = source
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
    return escaped.replace(/\n/g, '<br>')
  }
}
