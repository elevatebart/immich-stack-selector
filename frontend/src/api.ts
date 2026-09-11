export type Kind = 'burst' | 'format' | 'loose'
export type Status = 'pending' | 'reviewed' | 'skipped' | 'gone'

export interface Asset {
  id: string
  stack_id: string
  file_name: string
  created_at: string | null
  features: Record<string, number | null>
  score: number | null
}

export interface Stack {
  id: string
  kind: Kind
  captured_at: string | null
  primary_asset_id: string | null
  suggested_asset_id: string | null
  status: Status
  assets: Asset[]
}

export interface Stats {
  by_kind_status: { kind: Kind; status: Status; n: number }[]
  human_decisions: number
}

async function j<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, { headers: { 'content-type': 'application/json' }, ...init })
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`)
  return r.json() as Promise<T>
}

export const api = {
  stacks: (p: { status?: Status; kind?: Kind | ''; limit?: number }) =>
    j<Stack[]>(`/api/stacks?${new URLSearchParams(Object.entries(p).filter(([, v]) => v !== '' && v != null) as [string, string][])}`),
  decide: (id: string, chosen_asset_id: string, trash_others: boolean) =>
    j<Stack>(`/api/stacks/${id}/decision`, { method: 'POST', body: JSON.stringify({ chosen_asset_id, trash_others }) }),
  skip: (id: string) => j<Stack>(`/api/stacks/${id}/skip`, { method: 'POST' }),
  undo: (id: string) => j<Stack>(`/api/stacks/${id}/undo`, { method: 'POST' }),
  sync: (apply: boolean) => j<{ started: boolean }>(`/api/sync?apply=${apply}`, { method: 'POST' }),
  stats: () => j<Stats>('/api/stats'),
  thumb: (id: string, size: 'thumbnail' | 'preview' = 'thumbnail') => `/api/thumb/${id}?size=${size}`,
}
