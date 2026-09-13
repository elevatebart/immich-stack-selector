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

export type ProposalAction = 'keep' | 'create' | 'dissolve'
export type ProposalStatus = 'proposed' | 'applied' | 'rejected'

export interface Proposal {
  id: number
  action: ProposalAction
  status: ProposalStatus
  asset_ids: string[]
  primary_asset_id: string | null
  min_sim: number | null
  existing_stack_id: string | null
  replaces: string[]
  created_stack_id: string | null
  reason: string | null
  assets: { id: string; file_name: string | null; features: Record<string, number | null> }[]
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
  proposals: (p: { status?: ProposalStatus; action?: ProposalAction | ''; limit?: number }) =>
    j<Proposal[]>(`/api/proposals?${new URLSearchParams(Object.entries(p).filter(([, v]) => v !== '' && v != null) as [string, string][])}`),
  proposalStats: () => j<{ action: ProposalAction; status: ProposalStatus; n: number }[]>('/api/proposals/stats'),
  applyProposal: (id: number) => j<Proposal>(`/api/proposals/${id}/apply`, { method: 'POST' }),
  rejectProposal: (id: number) => j<Proposal>(`/api/proposals/${id}/reject`, { method: 'POST' }),
  applyAll: (action: ProposalAction | '') =>
    j<{ started: boolean; count: number }>(`/api/proposals/apply-all${action ? `?action=${action}` : ''}`, { method: 'POST' }),
  build: (taken_after: string) =>
    j<{ started: boolean }>(`/api/build${taken_after ? `?taken_after=${taken_after}` : ''}`, { method: 'POST' }),
  thumb: (id: string, size: 'thumbnail' | 'preview' = 'thumbnail') => `/api/thumb/${id}?size=${size}`,
}
