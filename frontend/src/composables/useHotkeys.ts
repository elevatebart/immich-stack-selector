import { onMounted, onUnmounted } from 'vue'

export type HotkeyMap = Record<string, (e: KeyboardEvent) => void>

export function useHotkeys(map: HotkeyMap) {
  const handler = (e: KeyboardEvent) => {
    const t = e.target as HTMLElement | null
    if (t && ['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName)) return
    if (e.metaKey || e.ctrlKey || e.altKey) return
    const fn = map[e.key] ?? map[e.key.toLowerCase()]
    if (fn) {
      e.preventDefault()
      fn(e)
    }
  }
  onMounted(() => window.addEventListener('keydown', handler))
  onUnmounted(() => window.removeEventListener('keydown', handler))
}
