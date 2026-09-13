# immich-stack-selector

Builds photo stacks in [Immich](https://immich.app) from timing, GPS and
visual similarity, puts the sharpest frame on top, and gives you a keyboard
driven page to keep that frame and trash the rest.

It talks to the public Immich API only. Runs as a single container on a NAS.

## What it does

Three passes, each one optional on its own:

1. **Build stacks.** Every image in the library is walked in capture order.
   Consecutive shots within `TIME_WINDOW_S` seconds (and, when both carry
   GPS, within `LOCATION_RADIUS_M` metres) form a candidate group. Frames in
   a group are embedded with CLIP and chained in upload order: a frame joins
   the most recent chain whose last frame it resembles (`SIM_THRESHOLD`
   cosine). Each chain of two or more frames becomes a stack. RAW+JPG pairs
   with the same file stem always land together. Several phones shooting the
   same event interleave without breaking chains.
2. **Pick the primary.** Every frame is scored on sharpness, face sharpness
   (using Immich's own face boxes), highlight and shadow clipping and
   exposure. Scores are standardised inside the stack and combined; the best
   frame is created as the stack primary, so Immich shows it in the
   timeline. Optional: the LAION aesthetic predictor on CLIP ViT-L/14.
3. **Review.** A web page shows one stack at a time with the pick
   preselected. `t` keeps it and trashes the siblings (Immich soft delete,
   `u` restores), a digit picks another frame first. Every decision is kept
   as a training pair; `python -m app.train` refits the ranker weights on
   them once you have a hundred or so.

Proposed stack changes are staged locally and diffed against what Immich
has. You see `keep`, `create` and `dissolve` cards with thumbnails, the
minimum pairwise similarity and the reason a stack fails, then apply them
one by one or in bulk. Nothing reaches Immich before you say so, except in
the container's timer mode, which only ever creates stacks for photos that
are not stacked yet.

## Deploy on a NAS

```bash
mkdir immich-stack-selector && cd immich-stack-selector
curl -LO https://raw.githubusercontent.com/elevatebart/immich-stack-selector/main/docker-compose.yml
curl -Lo .env https://raw.githubusercontent.com/elevatebart/immich-stack-selector/main/.env.example
# edit .env: IMMICH_URL, IMMICH_API_KEY
docker compose up -d
```

Open `http://nas:8000`. The `data/` folder next to the compose file keeps the
SQLite database, the thumbnail cache and the CLIP weights, so the first
build downloads the model once.

The API key needs these permissions: `stack.read`, `stack.create`,
`stack.update`, `stack.delete`, `asset.read`, `asset.view`, `face.read`,
and `asset.delete` for the trash and restore actions.

Timers, all in minutes and all disabled with `0`:

| variable | default | effect |
| --- | --- | --- |
| `SYNC_EVERY_MIN` | 360 | re-score every stack, refresh suggestions |
| `BUILD_EVERY_MIN` | 60 | stack unstacked photos taken in the last `BUILD_WINDOW_DAYS` |
| `BUILD_WINDOW_DAYS` | 14 | how far back the periodic build looks |

Prefer an explicit schedule (Synology Task Scheduler, cron)? Set the two
`*_EVERY_MIN` variables to `0` and run these as root on the host:

```bash
docker exec -w /app/backend immich-stack-selector python -m app.build --periodic 14
docker exec -w /app/backend immich-stack-selector python -m app.sync
```

Images are published to `ghcr.io/elevatebart/immich-stack-selector` for
`linux/amd64` and `linux/arm64` on every push to `main`. Uncomment
`build: .` in the compose file to build on the NAS instead.

## First run on an existing library

If you already have stacks (from immich-stack or by hand) the builder will
diff against them. On a fresh library or after unstacking everything, every
proposal is a `create`. From the proposals tab:

1. Click **rebuild proposals** with the date empty. The whole library is
   scanned; only the first run is slow because it embeds every candidate.
2. Cards are sorted worst first. Long chains with a low minimum similarity
   are the ones to eyeball: reject the ones that merged different scenes.
3. **apply all shown**, optionally filtered by action.
4. Switch to **review primaries** and work through the queue with the keys
   below.

Stacks whose consecutive frames are more than two windows apart are labelled
`loose`; the trash button is disabled on them as a guard. `format` stacks
(RAW+JPG) can never be trashed from this tool.

## Keys on the review page

| key | action |
| --- | --- |
| `1`..`9`, arrows | select a frame |
| `enter` | set selected frame as primary |
| `t` | set primary and trash the other frames (burst stacks only) |
| `s` | skip |
| `u` | undo last decision, restores trashed frames |

## Tuning

| variable | default | notes |
| --- | --- | --- |
| `TIME_WINDOW_S` | 120 | max gap between consecutive shots in a candidate group |
| `LOCATION_RADIUS_M` | 30 | phone GPS jitters by 10 to 30 m, keep this above 10 |
| `SIM_THRESHOLD` | 0.85 | 0.90 splits reframed shots of the same scene, 0.80 starts merging different subjects |
| `CLUSTER_MODE` | `chain` | `complete` requires every frame to match every other, tighter stacks, no cross device chains |
| `EMBED_MODEL` | `ViT-B-32/openai` | any open_clip `arch/pretrained` pair |

Embeddings and features are cached, so changing thresholds and rebuilding
takes a couple of minutes on a 20k photo library.

## Local development

```bash
cp .env.example .env
cd backend && python3 -m venv .venv && .venv/bin/pip install -e '.[aesthetic]'
.venv/bin/python -m app.build --taken-after 2026-08-01    # dry run, fills proposals
.venv/bin/python -m app.sync                              # score existing stacks
.venv/bin/uvicorn app.main:app --reload --port 8000 --app-dir .
cd ../frontend && npm install && npm run dev              # http://localhost:5173, proxies /api
```

`npm run build` writes `frontend/dist`, which the API serves at `/`.

Layout: `backend/app` holds the FastAPI app (`main.py`), the Immich client,
the SQLite layer, the stack builder (`build.py`, `group.py`) and the scorer
(`scoring/`). `frontend/src` is a small Vue 3 app with two views.

## Caveats

- Chain mode can drift: a long burst where each frame resembles the previous
  one can end with frames that no longer resemble the first. The card shows
  the minimum pairwise similarity for exactly this reason.
- If immich-stack still runs in cron mode with `REPLACE_STACKS`, it will
  rebuild stacks with its own parent rule and undo the primaries. Stop it.
- CLIP measures semantic similarity, not pixel identity. Two different
  portraits of the same person at the same table can score above 0.85.

## License

MIT
