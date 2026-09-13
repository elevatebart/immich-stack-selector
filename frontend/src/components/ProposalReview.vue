<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { api, type Proposal, type ProposalAction, type ProposalStatus } from '../api'

const status = ref<ProposalStatus>('proposed')
const action = ref<ProposalAction | ''>('')
const rows = ref<Proposal[]>([])
const stats = ref<{ action: ProposalAction; status: ProposalStatus; n: number }[]>([])
const busy = ref<number | null>(null)
const error = ref('')
const since = ref('')

const summary = computed(() =>
  (['keep', 'create', 'dissolve'] as ProposalAction[])
    .map((a) => `${a} ${stats.value.filter((s) => s.action === a && s.status === 'proposed').reduce((n, s) => n + s.n, 0)}`)
    .join(' · '),
)

async function load() {
  error.value = ''
  try {
    ;[rows.value, stats.value] = await Promise.all([api.proposals({ status: status.value, action: action.value, limit: 100 }), api.proposalStats()])
  } catch (e) {
    error.value = String(e)
  }
}

async function act(p: Proposal, fn: (id: number) => Promise<Proposal>) {
  busy.value = p.id
  error.value = ''
  try {
    const updated = await fn(p.id)
    rows.value = rows.value.filter((r) => r.id !== updated.id)
    stats.value = await api.proposalStats()
  } catch (e) {
    error.value = String(e)
  } finally {
    busy.value = null
  }
}

async function applyAll() {
  const n = stats.value.filter((s) => s.status === 'proposed' && (!action.value || s.action === action.value)).reduce((a, s) => a + s.n, 0)
  if (!confirm(`Apply ${n} ${action.value || 'proposed'} proposals to Immich? Dissolve and replace unstack photos; nothing is trashed.`)) return
  await api.applyAll(action.value)
  setTimeout(load, 1500)
}

async function rebuild() {
  if (!confirm(`Rebuild proposals${since.value ? ` for photos taken after ${since.value}` : ' for the whole library'}? Existing unapplied proposals are replaced.`)) return
  await api.build(since.value)
  error.value = 'build started in the background, reload in a bit'
}

function label(p: Proposal) {
  if (p.action === 'keep') return 'keep existing stack'
  if (p.action === 'dissolve') return `dissolve stack: ${p.reason ?? 'fails time, GPS or similarity'}`
  return p.replaces.length ? `create, replacing ${p.replaces.length} stack${p.replaces.length > 1 ? 's' : ''}` : 'create new stack'
}

onMounted(load)
</script>

<template>
  <div class="proposals">
    <div class="toolbar">
      <select v-model="status" @change="load">
        <option value="proposed">proposed</option>
        <option value="applied">applied</option>
        <option value="rejected">rejected</option>
      </select>
      <select v-model="action" @change="load">
        <option value="">all actions</option>
        <option value="create">create</option>
        <option value="dissolve">dissolve</option>
        <option value="keep">keep</option>
      </select>
      <span class="muted">{{ summary }}</span>
      <span class="spacer" />
      <input v-model="since" type="date" title="only scan photos taken after" />
      <button @click="rebuild">rebuild proposals</button>
      <button class="danger" :disabled="status !== 'proposed'" @click="applyAll">apply all shown</button>
    </div>
    <p v-if="error" class="error">{{ error }}</p>
    <p v-if="!rows.length && !error" class="empty">No proposals here. Run <code>python -m app.build</code> or click rebuild.</p>
    <article v-for="p in rows" :key="p.id" class="card" :class="p.action">
      <header>
        <span class="badge" :class="p.action">{{ p.action }}</span>
        <span>{{ label(p) }}</span>
        <span v-if="p.min_sim != null" class="muted">min sim {{ p.min_sim.toFixed(3) }}</span>
        <span class="muted">{{ p.asset_ids.length }} frames</span>
        <span class="spacer" />
        <template v-if="p.status === 'proposed'">
          <button :disabled="busy === p.id" @click="act(p, api.rejectProposal)">reject</button>
          <button class="primary" :disabled="busy === p.id" @click="act(p, api.applyProposal)">apply</button>
        </template>
        <span v-else class="muted">{{ p.status }}</span>
      </header>
      <div class="strip">
        <figure v-for="a in p.assets" :key="a.id" :class="{ primary: a.id === p.primary_asset_id }">
          <img :src="api.thumb(a.id)" loading="lazy" />
          <figcaption>
            <span v-if="a.id === p.primary_asset_id" class="tag">primary</span>
            {{ a.file_name ?? a.id.slice(0, 8) }}
          </figcaption>
        </figure>
      </div>
    </article>
  </div>
</template>

<style scoped>
.proposals { flex: 1; min-height: 0; overflow-y: auto; padding: 12px 16px; display: flex; flex-direction: column; gap: 12px; }
.toolbar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.spacer { flex: 1; }
.card { border: 1px solid var(--border); border-radius: 8px; background: var(--panel); padding: 10px 12px; }
.card.dissolve { border-color: #6b3a3a; }
.card.create { border-color: #3a5f6b; }
header { display: flex; gap: 10px; align-items: center; font-size: 13px; margin-bottom: 8px; }
.badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; text-transform: uppercase; background: #333; }
.badge.create { background: #1f5c6b; }
.badge.dissolve { background: #6b2a2a; }
.badge.keep { background: #2f5f2f; }
.strip { display: flex; gap: 8px; overflow-x: auto; }
figure { margin: 0; flex: 0 0 auto; display: flex; flex-direction: column; gap: 4px; }
figure img { height: 120px; width: auto; border-radius: 6px; border: 2px solid transparent; display: block; }
figure.primary img { border-color: var(--accent); }
figcaption { font-size: 11px; color: var(--muted); max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tag { color: var(--accent); margin-right: 4px; }
.muted { color: var(--muted); font-size: 12px; }
.error { color: #e08080; }
.empty { color: var(--muted); text-align: center; margin-top: 40px; }
button.danger { border-color: #6b2a2a; }
</style>
