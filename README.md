# immich-stack-selector

Scores every frame of your Immich stacks, sets the best one as the stack
primary, and gives you a keyboard-driven UI to confirm the pick and trash
the rest. Every review you make becomes a training pair for the ranker.

Works on top of stacks created by [immich-stack](https://github.com/majorfi/immich-stack)
or by hand. Only talks to the public Immich API.

### How scoring works

1. `python -m app.sync` pulls all stacks, classifies each one:
   `burst` (same extension, frames within `TIME_WINDOW_S` of each other),
   `format` (mixed extensions, typically RAW+JPG), `loose`.
2. For each frame it downloads the `preview` thumbnail and computes
   sharpness (Laplacian variance), face sharpness on Immich's own face boxes,
   clipping and exposure. With `USE_AESTHETIC=1` it adds the LAION aesthetic
   score (CLIP ViT-L/14). Features are cached in SQLite.
3. Features are z-scored within the stack and combined with a weight vector.
   Default weights are hand-set. `python -m app.train` refits them by
   logistic regression on your review decisions (chosen vs rejected sibling).
4. `--apply` writes the suggested primary to Immich for unreviewed burst
   stacks. Without it, suggestions only show up in the UI.

Trashing is refused for `format` stacks so you never lose a RAW. Trash uses
Immich's soft delete, `undo` restores from trash and resets the primary.

### Stack builder (work in progress)

`python -m app.build` replaces immich-stack's grouping. It lists every image
via `POST /search/metadata`, chains consecutive shots into candidate groups
when they are within `TIME_WINDOW_S` and, when both carry GPS, within
`LOCATION_RADIUS_M`, embeds the previews with CLIP (`EMBED_MODEL`) and keeps
connected components whose pairwise cosine is at least `SIM_THRESHOLD`.
RAW+JPG pairs with the same stem are always linked. Each cluster's primary
is the frame the ranker scores highest.

Proposals are diffed against the stacks Immich has today and stored locally
as `keep`, `create` (replacing overlapping stacks) or `dissolve`. Nothing is
written until `--apply`. `--taken-after` / `--taken-before` limit the scan;
existing stacks outside the window are left alone.

```bash
.venv/bin/python -m app.build --taken-after 2026-08-01   # dry run, fills the proposals table
.venv/bin/python -m app.build --taken-after 2026-08-01 --apply
```

Known gap: clusters are single-linkage, so a chain A~B~C can pull in an A/C
pair well below the threshold (the dry run reported one 9-frame chain with a
min pairwise sim of 0.82 against a 0.90 threshold). The `min_sim` column
exposes this; a UI to review proposals before applying is not built yet.

### Run

```bash
cp .env.example .env   # set IMMICH_URL and IMMICH_API_KEY
cd backend && python3 -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python -m app.sync            # score, no writes to Immich
.venv/bin/python -m app.sync --apply    # also set primaries on burst stacks
.venv/bin/uvicorn app.main:app --reload --port 8000
cd ../frontend && npm install && npm run dev   # http://localhost:5173
```

Production: `npm run build` in `frontend/`, the FastAPI app serves `dist/` at `/`.

Optional aesthetic model: `.venv/bin/pip install -e '.[aesthetic]'` and `USE_AESTHETIC=1`.
First run downloads CLIP ViT-L/14 (~1.7 GB) and the predictor head.

### Review UI

| key | action |
| --- | --- |
| `1`..`9`, arrows | select a frame |
| `enter` | set selected frame as primary |
| `t` | set primary and trash the other frames (burst stacks only) |
| `s` | skip |
| `u` | undo last decision (restores trashed frames) |

### Caveats

- immich-stack in `RUN_MODE=cron` with `REPLACE_STACKS` may rebuild stacks
  and re-apply its own parent rule, overwriting your primary. Run it once,
  or chain `python -m app.sync --apply` after it.
- `sync` marks stacks that disappeared from Immich as `gone` rather than
  deleting local rows, so decisions stay available for training.
- The ranker needs on the order of 100 human decisions before it beats the
  hand-set weights. Retrain with `python -m app.train`, then rescore with
  `python -m app.sync`.

### Docker (NAS deployment)

The image bundles the UI, the API and CPU-only torch. Model weights, the
SQLite database and the thumbnail cache live on the `/data` volume, so the
first `build` downloads CLIP once and keeps it.

```bash
cp .env.example .env            # fill IMMICH_URL and IMMICH_API_KEY
docker compose up -d            # pulls ghcr.io/elevatebart/immich-stack-selector
docker compose exec stack-selector python -m app.build --app-dir backend   # propose stacks
```

`SYNC_EVERY_MIN` re-scores stacks on a timer (default six hours in the
compose file). The builder and the bulk apply stay manual: run them from
the proposals tab or with `docker compose exec stack-selector python -c
"from app.build import propose; propose(None, None)"`. The GitHub workflow
publishes `linux/amd64` and `linux/arm64` images on every push to `main`;
to build on the NAS itself, uncomment `build: .` in the compose file.
