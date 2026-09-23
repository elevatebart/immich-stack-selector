# Architecture

How photos become stacks, how a stack gets its primary, and where the
numbers come from. File paths are relative to `backend/app/`.

## Data flow

```
Immich API ──search/metadata──▶ candidate groups ──CLIP──▶ chains ──▶ proposals ──apply──▶ POST /stacks
     │                             (group.py)              (group.py)   (build.py)
     └──GET /stacks────────────▶ per-frame features ──▶ ranker ──▶ suggested primary ──PUT /stacks/{id}
                                 (scoring/technical.py)  (scoring/ranker.py)   (sync.py)
                                                                  ▲
                                            review decisions ─────┘ (train.py, logistic fit)
```

All state lives in one SQLite file (`db.py`). Immich is only ever read
through its public API and written through `POST /stacks`,
`PUT /stacks/{id}`, `DELETE /stacks/{id}`, `DELETE /assets` (soft delete)
and `POST /trash/restore/assets`.

## 1. Stack builder

### Candidate groups (`group.py: candidate_groups`)

Every image of the key owner is listed once, oldest first, with EXIF.
Frames are walked per owner in capture order (`dateTimeOriginal`, falling
back to `fileCreatedAt`). A frame extends the current group when

- it was taken at most `TIME_WINDOW_S` seconds after the previous frame, and
- both frames have GPS and lie within `LOCATION_RADIUS_M` metres
  (haversine), or at least one has no GPS.

A gap in time or space closes the group. Groups of one frame are dropped.
This pass is cheap and exists only to bound the pairwise work of the next
one: similarity is never computed across groups.

### Embeddings (`scoring/embed.py`)

Each frame's Immich `preview` thumbnail (long side about 1440 px) goes
through the CLIP ViT-B/32 image tower with OpenAI weights, loaded via
`open_clip` (`EMBED_MODEL=ViT-B-32/openai`). The standard preprocessing
resizes to 224 px and centre-crops. Output is a 512-d vector, L2
normalised, stored as float32 bytes in the `embeddings` table keyed by
asset id and model name. Nothing is recomputed on a threshold change.

Similarity between two frames is the cosine, which after normalisation is
the dot product. Same-stem RAW+JPG pairs are forced to 1.0 before
clustering (`format_pairs`).

Why CLIP and not perceptual hashing: burst frames differ by a reframe, a
blink or a step to the side, which moves a pHash a lot and a CLIP embedding
very little. Why ViT-B/32 and not a bigger model: the group already bounds
the candidates to the same minute and place, so the model only needs to
tell "same scene" from "different scene", and B/32 runs on a NAS CPU.

### Clustering (`group.py: chain_clusters`, `complete_clusters`)

Two algorithms, selected by `CLUSTER_MODE`.

`chain` (default). Frames in a group are visited in Immich upload order
(`createdAt`), which keeps one device's frames consecutive. Each frame is
compared to the last frame of every open chain, newest chain first, and
appended to the first one at or above `SIM_THRESHOLD`; otherwise it opens a
chain. This is single linkage with a recency bias. It lets two phones at
the same event interleave without splitting either burst, at the price of
drift: frame N can end far from frame 1. The stored `min_sim` (lowest
pairwise cosine inside the cluster) exposes the drift on each proposal.

`complete`. A frame joins a cluster only if it clears the threshold against
every member. No drift, tighter stacks, more splits when devices interleave.

After either pass, forced RAW+JPG pairs that landed in different clusters
are merged. Clusters of one are dropped.

### Identity split (`group.py: split_by_identity`, `scoring/faces.py`)

CLIP embeds the whole frame, so portraits of different people against one
backdrop clear the threshold. Each cluster is re-split on who is in it.
Immich's face boxes are re-detected with SCRFD (buffalo_l `det_10g`) inside a
2x crop to get five landmarks, aligned to the ArcFace template and embedded
with `w600k_r50`, the same models Immich runs. Vectors are cached in
`face_embeddings`; an asset with no Immich faces yet is retried next run.

A face is a main subject when its box side is at least `FACE_MIN_SIZE` of
the short image side, SCRFD scores it 0.6 or more and it is frontal: the
nose sits within 0.25 eye distances of the eye midpoint. Two frames are
different subjects when both have main faces and no main face of either
matches any face of the other at `FACE_THRESHOLD` cosine. Comparing main to
all faces keeps a panning group shot together. A shared Immich person id
short-circuits to "same"; disjoint ids prove nothing, since Immich files one
person under two ids often enough (glasses, goggles).

Frames are assigned in cluster order to the first part where they conflict
with at most half of the comparable members, otherwise they open a new part.
One blurry or dark face therefore cannot pull a frame out of a burst, but a
two-frame cluster splits on a single conflict. Parts of one are dropped.

Calibration on the author's library, 488 clusters over 2024 to 2026: 2
correct splits (different people against one backdrop), 1 frame wrongly
left out of a stack. Rules tried and dropped: a single conflict splits
(3 of 5 splits wrong), disjoint person ids split (15 of 410 real stacks).

### Proposals (`build.py: propose`)

Each cluster is scored by the ranker (section 2) so the primary is known
before the stack exists. Clusters are then diffed against `GET /stacks`:

| action | condition |
| --- | --- |
| `keep` | an Immich stack has exactly this asset set |
| `create` | no such stack; `replaces` lists existing stacks that overlap it |
| `dissolve` | an existing stack in the scanned window matches no cluster |

Dissolve proposals carry a `reason`: the widest gap in time, the GPS spread,
or the lowest pairwise similarity, whichever rule fails first. Applying a
`create` deletes the replaced stacks and calls `POST /stacks` with the
primary first, which is how Immich picks the cover. `--periodic DAYS` only
applies `create` rows with no `replaces`, so an unattended run can never
undo a human decision.

## 2. Primary selection

### Features (`scoring/technical.py`)

Computed on the `preview` thumbnail, downscaled to 1024 px and converted to
8-bit grey. Cached per asset in `feature_cache`, versioned with
`FEATURE_VERSION`.

| feature | definition | why |
| --- | --- | --- |
| `sharpness` | log(1 + variance of the 4-neighbour Laplacian) | blur is the main difference inside a burst |
| `face_sharpness` | same, averaged over Immich's face boxes (`GET /faces`) | a sharp background with a blurred face still loses |
| `face_area` | face pixels / image pixels | prefers the frame where people are larger |
| `shadow_clip` | fraction of pixels below 8 | crushed blacks |
| `highlight_clip` | fraction of pixels above 247 | blown highlights |
| `lum_dev` | minus the distance of mean luminance from 0.45 | avoids the darkest and brightest frame of an exposure bracket |
| `aesthetic` | optional, see below | composition, when frames differ by more than sharpness |

The log on the Laplacian variance makes a 2x blur difference count the same
at ISO 100 and ISO 6400.

### Optional aesthetic head (`scoring/aesthetic.py`, `USE_AESTHETIC=1`)

The LAION "improved aesthetic predictor": a 5-layer MLP (768 to 1024, 128,
64, 16, 1) trained on SAC, LAION-Logos and AVA ratings, applied to the
L2-normalised CLIP ViT-L/14 image embedding. Output is a 1 to 10 score. It
needs a second, larger CLIP (about 1.7 GB), so it is off by default. It is
weak at telling burst frames apart and is meant for stacks that differ by
composition.

### Ranker (`scoring/ranker.py`)

Per stack, each feature is z-scored across the stack's frames. Missing
values (no face) are set to the stack mean and contribute nothing. The
standard deviation is floored per feature (`STD_FLOOR`) so two frames that
differ by noise are not scored as a landslide. The score is a weighted sum
of the z-scores; the highest frame becomes `suggested_asset_id`.

Default weights, hand set:

| feature | weight |
| --- | --- |
| `face_sharpness` | +1.5 |
| `sharpness` | +1.0 |
| `aesthetic` | +1.0 |
| `face_area` | +0.3 |
| `lum_dev` | +0.3 |
| `shadow_clip` | -0.3 |
| `highlight_clip` | -0.5 |

Because features are standardised within the stack, the score is a
relative judgement. It says nothing about whether a frame is good, only
which one of its siblings is best.

### Learning from reviews (`train.py`)

Every pick in the review page stores the chosen frame and its rejected
siblings. `python -m app.train` builds one training vector per
chosen/rejected pair, `z(chosen) - z(rejected)`, mirrors the set so the
bias is zero, and fits the weight vector by L2-regularised logistic
regression with plain gradient descent (500 epochs, lr 0.1, l2 1.0),
starting from the default weights. The result is written to
`data/ranker.json` and picked up by the next `sync`. Below about 100 pairs
the fit is noisy and the command says so.

### Stack kinds (`classify.py`)

Every stack is labelled after scoring:

- `format`: more than one extension, or any RAW extension. RAW+JPG pairs.
  The trash action is refused for these.
- `burst`: one extension and no two consecutive frames more than
  `2 * TIME_WINDOW_S` apart (a chain may have skipped a middle frame).
- `loose`: anything else, typically long event chains. Trash is disabled.

## 3. Review loop

The API (`main.py`) serves the built Vue app and lists pending stacks newest
first, frames inside a stack sorted by score. A decision writes the primary with `PUT /stacks/{id}`,
optionally soft-deletes the siblings with `DELETE /assets`, and appends a
row to `decisions`. Undo restores from trash and puts the previous primary
back. Decisions are the training set of section 2.

## Storage

| table | content |
| --- | --- |
| `stacks`, `assets` | mirror of Immich stacks with kind, status, score and suggestion |
| `feature_cache` | technical features per asset, survives stack rebuilds |
| `embeddings` | CLIP vectors per asset and model |
| `face_embeddings` | ArcFace vectors, detector score, size and person id per face |
| `proposals` | builder output with action, status, `min_sim`, `reason` |
| `decisions` | human and automatic primary changes, trashed ids for undo |

Thumbnails are cached under `CACHE_DIR/{size}/{asset_id}.jpg`, model
weights under `CACHE_DIR/models`. Deleting `data/` loses decisions and the
trained weights; everything else is rebuilt from Immich.

## Known limits

- Chain drift, discussed above. Watch `min_sim` under 0.8.
- CLIP similarity is semantic. A burst with a large exposure change can fall
  under the threshold. Different portraits at the same table are caught by
  the identity split only when faces are large enough; a series of wide
  shots or people seen from behind still stacks.
- The buffalo_l weights are under the InsightFace licence, non-commercial
  use only. `FACE_SPLIT=0` skips the download.
- Sharpness is measured on a 1024 px preview. Micro blur that only shows at
  100 % is invisible to it.
- Face boxes come from Immich's own detector. Frames Immich has not
  processed yet have `face_sharpness` unset and are ranked on the rest.
