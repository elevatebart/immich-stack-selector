<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type Kind, type Stack, type Stats, type Status } from './api'
import ProposalReview from './components/ProposalReview.vue'
import StackReview from './components/StackReview.vue'
import { useHotkeys } from './composables/useHotkeys'

const queue = ref<Stack[]>([])
const history = ref<Stack[]>([])
const selected = ref(0)
const busy = ref(false)
const error = ref('')
const stats = ref<Stats | null>(null)
const kind = ref<Kind | ''>('burst')
const status = ref<Status>('pending')
const view = ref<'review' | 'proposals'>(new URLSearchParams(location.search).get('view') === 'proposals' ? 'proposals' : 'review')

const current = computed(() => queue.value[0])
const pendingCount = computed(() =>
  stats.value?.by_kind_status.filter((r) => r.status === 'pending' && (!kind.value || r.kind === kind.value))
    .reduce((n, r) => n + r.n, 0) ?? 0,
)

async function load() {
  busy.value = true
  error.value = ''
  try {
    ;[queue.value, stats.value] = await Promise.all([api.stacks({ status: status.value, kind: kind.value, limit: 50 }), api.stats()])
    selectSuggested()
  } catch (e) {
    error.value = String(e)
  } finally {
    busy.value = false
  }
}

function selectSuggested() {
  const st = current.value
  if (!st) return
  const i = st.assets.findIndex((a) => a.id === st.suggested_asset_id)
  selected.value = i >= 0 ? i : 0
}

async function act(fn: (st: Stack) => Promise<Stack>) {
  const st = current.value
  if (!st || busy.value) return
  busy.value = true
  error.value = ''
  try {
    const updated = await fn(st)
    history.value.unshift(updated)
    queue.value.shift()
    if (queue.value.length < 5) await load()
    else selectSuggested()
    stats.value = await api.stats()
  } catch (e) {
    error.value = String(e)
  } finally {
    busy.value = false
  }
}

const pick = (trash: boolean) =>
  act((st) => api.decide(st.id, st.assets[selected.value].id, trash && st.kind === 'burst'))
const skip = () => act((st) => api.skip(st.id))

async function undo() {
  const last = history.value[0]
  if (!last || busy.value) return
  busy.value = true
  try {
    const restored = await api.undo(last.id)
    history.value.shift()
    queue.value.unshift(restored)
    selectSuggested()
    stats.value = await api.stats()
  } catch (e) {
    error.value = String(e)
  } finally {
    busy.value = false
  }
}

async function sync() {
  await api.sync(false)
  error.value = 'sync started in the background, reload in a bit'
}

const move = (d: number) => {
  const n = current.value?.assets.length ?? 0
  if (n) selected.value = (selected.value + d + n) % n
}

// hotkeys only make sense on the review screen
const onReview = (fn: () => void) => () => { if (view.value === 'review') fn() }
useHotkeys(Object.fromEntries(Object.entries({
  ArrowLeft: () => move(-1),
  ArrowRight: () => move(1),
  Enter: () => pick(false),
  t: () => pick(true),
  s: skip,
  u: undo,
  ...Object.fromEntries(
    Array.from({ length: 9 }, (_, i) => [String(i + 1), () => { if ((current.value?.assets.length ?? 0) > i) selected.value = i }]),
  ),
}).map(([k, fn]) => [k, onReview(fn as () => void)])))

onMounted(load)
</script>

<template>
  <header class="bar">
    <strong>immich stack selector</strong>
    <nav class="tabs">
      <button :class="{ active: view === 'review' }" @click="view = 'review'">review primaries</button>
      <button :class="{ active: view === 'proposals' }" @click="view = 'proposals'">stack proposals</button>
    </nav>
    <template v-if="view === 'review'">
    <select v-model="kind" @change="load">
      <option value="burst">burst</option>
      <option value="format">format (RAW+JPG)</option>
      <option value="loose">loose (fails time window)</option>
      <option value="">all kinds</option>
    </select>
    <select v-model="status" @change="load">
      <option value="pending">pending</option>
      <option value="skipped">skipped</option>
      <option value="reviewed">reviewed</option>
    </select>
    <span class="muted">{{ pendingCount }} pending, {{ stats?.human_decisions ?? 0 }} decisions logged</span>
    <span class="grow" />
    <button @click="sync">sync</button>
    <button :disabled="!history.length" @click="undo">undo <kbd>u</kbd></button>
    </template>
  </header>

  <ProposalReview v-if="view === 'proposals'" />

  <p v-if="error && view === 'review'" class="error">{{ error }}</p>

  <main v-if="view === 'review' && current">
    <StackReview :stack="current" :selected="selected" @select="selected = $event" />
    <footer class="bar actions">
      <span class="muted">{{ queue.length }} in queue</span>
      <span class="grow" />
      <button @click="skip">skip <kbd>s</kbd></button>
      <button @click="pick(false)">set as primary <kbd>enter</kbd></button>
      <button class="danger" :disabled="current.kind !== 'burst'" @click="pick(true)">
        primary + trash {{ current.assets.length - 1 }} others <kbd>t</kbd>
      </button>
    </footer>
  </main>
  <p v-else-if="view === 'review' && !busy" class="muted empty">Nothing to review. Run <code>python -m app.sync</code> or change the filters.</p>
</template>

<style scoped>
.bar { display: flex; align-items: center; gap: 12px; padding: 10px 16px; border-bottom: 1px solid var(--border); background: var(--panel); flex: 0 0 auto; }
main { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.actions { border-top: 1px solid var(--border); border-bottom: 0; }
.grow { flex: 1; }
.tabs { display: flex; gap: 2px; }
.tabs button { opacity: 0.6; }
.tabs button.active { opacity: 1; border-color: var(--accent); }
.muted { color: var(--muted); }
.error { color: var(--warn); padding: 8px 16px; margin: 0; }
.empty { padding: 48px; text-align: center; }
</style>
