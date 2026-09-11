<script setup lang="ts">
import { computed } from 'vue'
import { api, type Asset, type Stack } from '../api'

const props = defineProps<{ stack: Stack; selected: number }>()
const emit = defineEmits<{ select: [index: number] }>()

const chosen = computed(() => props.stack.assets[props.selected])

const badges = (a: Asset) => [
  a.id === props.stack.primary_asset_id ? 'primary' : '',
  a.id === props.stack.suggested_asset_id ? 'suggested' : '',
].filter(Boolean)

const fmt = (v: number | null | undefined, d = 2) => (v == null ? '-' : v.toFixed(d))
const when = (iso: string | null) => (iso ? new Date(iso).toLocaleString() : '')
</script>

<template>
  <section class="review">
    <div class="preview">
      <img :src="api.thumb(chosen.id, 'preview')" :alt="chosen.file_name" />
      <div class="meta">
        <span>{{ chosen.file_name }}</span>
        <span class="muted">{{ when(chosen.created_at) }}</span>
        <span class="muted">
          score {{ fmt(chosen.score) }} · sharp {{ fmt(chosen.features.sharpness) }}
          · face {{ fmt(chosen.features.face_sharpness) }} ({{ chosen.features.faces ?? 0 }})
          <template v-if="chosen.features.aesthetic != null"> · aesthetic {{ fmt(chosen.features.aesthetic) }}</template>
        </span>
        <span class="kind" :data-kind="stack.kind">{{ stack.kind }}</span>
      </div>
    </div>

    <div class="strip">
      <button
        v-for="(a, i) in stack.assets"
        :key="a.id"
        class="card"
        :class="{ active: i === selected }"
        @click="emit('select', i)"
      >
        <img :src="api.thumb(a.id)" :alt="a.file_name" loading="lazy" />
        <span class="num">{{ i + 1 }}</span>
        <span class="score">{{ fmt(a.score) }}</span>
        <span v-for="b in badges(a)" :key="b" class="badge" :data-badge="b">{{ b }}</span>
      </button>
    </div>
  </section>
</template>

<style scoped>
.review { flex: 1; min-height: 0; display: flex; flex-direction: column; gap: 12px; padding: 12px 16px; }
.preview { flex: 1; min-height: 0; display: flex; flex-direction: column; align-items: center; gap: 8px; }
.preview img { flex: 1; min-height: 0; max-width: 100%; object-fit: contain; border-radius: 8px; background: #000; }
.meta { display: flex; gap: 16px; flex-wrap: wrap; align-items: center; }
.muted { color: var(--muted); }
.kind { border: 1px solid var(--border); border-radius: 999px; padding: 0 10px; font-size: 12px; }
.kind[data-kind='format'] { border-color: var(--warn); color: var(--warn); }
.strip { flex: 0 0 auto; display: flex; gap: 8px; overflow-x: auto; padding: 4px 0 8px; }
.card { position: relative; flex: 0 0 auto; padding: 0; border: 2px solid var(--border); border-radius: 8px; overflow: hidden; background: #000; }
.card.active { border-color: var(--accent); }
.card img { display: block; height: 140px; width: auto; }
.num, .score, .badge { position: absolute; font-size: 11px; padding: 1px 6px; border-radius: 4px; background: rgba(0,0,0,.7); }
.num { top: 4px; left: 4px; }
.score { bottom: 4px; right: 4px; font-family: ui-monospace, monospace; }
.badge { bottom: 4px; left: 4px; }
.badge[data-badge='primary'] { color: var(--ok); }
.badge[data-badge='suggested'] { color: var(--accent); bottom: 22px; }
</style>
