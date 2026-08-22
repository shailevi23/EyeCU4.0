# Current TeamAssigner — exact behavior audit

Source read: `trackers/team_assigner.py` (full file, 251 lines), its call site
in `full_pipeline.py` (`self.team_assigner.assign_teams_to_tracks(frames,
tracks)`, step 7 of `_process_video_advanced`), and its three direct tests in
`tests/test_tracking_contract.py` (`test_referees_are_never_assigned_a_team`,
`test_goalkeepers_are_never_assigned_a_team`,
`test_outfield_players_do_get_a_team`). No code changed in this pass.

## A. ROI

Per observation, `get_player_color(frame, bbox)` crops the player's own bbox
out of the frame, then takes a **chest sub-region of that crop**:

```
y: [0.15h, 0.50h]   x: [0.25w, 0.75w]
```

(`h, w` = the player crop's own height/width — fractions of the *player box*,
not the full frame.) This is NOT the full bbox and NOT a fixed top-half crop.

**Dead parameter:** `TeamAssigner.__init__(..., use_top_half=True)` is
accepted and stored (`self.use_top_half`) but is **never read anywhere in the
class** — `get_player_color` always uses the chest-ROI fractions above
regardless of its value. Anyone tuning `use_top_half` today would have no
effect.

## B. Color representation

Raw **BGR pixel median** — `np.median(torso.reshape(-1, 3), axis=0)` — a
3-vector, computed per observation. No HSV/Lab conversion, no histogram, no
mean (median chosen specifically to resist outlier pixels within the ROI).
KMeans (Phase D below) is fit and predicts directly on these 3-D BGR median
vectors — there is no separate "descriptor" object anywhere in this file.

## C. Filtering

Three independent, stacked filters, each with a documented fallback if it
would leave too little data:

1. **Overlap exclusion** (`_collect_color_samples(..., skip_overlaps=True)`):
   an observation is dropped if any *other* player's box covers more than
   25% of *this* player's own box area (`_bbox_overlap_ratio`, directional).
   Rationale in-code: an occluded/adjacent crop mixes in a neighbor's jersey
   color. Used both when pooling samples to fit the two team colors, and
   when aggregating each player's own color.

2. **Dark-outlier exclusion at fit time only** (`_fit_team_colors`): before
   fitting KMeans on the *pooled* sample set, any sample with
   `max(B,G,R) < 100` is dropped — intended to keep referees/goalkeepers
   (dark kit) from hijacking one of the two team clusters. If this leaves
   fewer than `num_teams` samples, the filter is abandoned and everything is
   used instead.

3. **Bright-subset preference at per-player aggregation** (inside
   `assign_teams_to_tracks`'s per-player loop): once a player's own sample
   list is chosen (clean/non-overlapping if available, else all), if at
   least `max(3, 30%)` of that player's samples have `max(B,G,R) >= 150`,
   only that bright subset is median-ed; otherwise every sample is used. This
   is why a genuinely dark-kit player (the referee) keeps their real dark
   color instead of being forced toward a nonexistent bright subset.

Two more fallback cascades exist:
- If the overlap-filtered pooled sample count is `< num_teams * 3`, fitting
  re-runs on the unfiltered (overlaps included) sample pool.
- If the pooled sample count is `< num_teams` even then, `_fit_team_colors`
  is skipped entirely — `self.kmeans` stays `None` and every player defaults
  to `team_id = 1`.

## D. Team prototype fitting

`KMeans(n_clusters=2, init='k-means++', n_init=10)` fit once per
`assign_teams_to_tracks()` call, on chest-color samples pooled from **only
the first `min(20, len(frames))` frames** of the (already skip_frames-
subsampled) processed sequence — not the whole match. Cluster centers are
stored as `self.team_colors[1]`, `self.team_colors[2]` (BGR ints, used only
for the `print()` log line — not for on-screen drawing, which uses the fixed
`display_colors = {1: (255,60,60), 2: (60,60,255)}` blue/red pair instead).

## E. Per-frame vs per-track behavior — two DIFFERENT policies exist, only one is live

The class contains two separate mechanisms with different semantics, and
only one of them is reachable from the production pipeline:

- **`get_player_team(frame, bbox, player_id)`** — classic
  first-observation-locks-forever: checks `player_id in self.player_team_dict`
  and returns the cached value immediately if present, otherwise classifies
  the single frame passed in and caches it forever. **This method is dead
  code in the active pipeline** — grep across the repo shows it is never
  called from `full_pipeline.py`, `football_tracker.py`, or any test; the
  only other `get_player_team` in the codebase is an unrelated same-named
  method on a different class in `experimental/event_detection/` (excluded,
  unintegrated code).
- **`assign_teams_to_tracks(frames, tracks)`** — the ONLY method the
  production pipeline calls (`full_pipeline.py` step 7). For each
  `player_id` that appears anywhere in the sequence, it aggregates **every
  frame that player_id appears in, across the ENTIRE processed sequence**
  (preferring non-overlapping and then bright samples per the filters
  above), takes the **median** BGR of that pool, and predicts ONE team from
  that single aggregate vector. This decision is written into
  `self.player_team_dict[player_id]` **unconditionally** (no
  check-before-write) every time `assign_teams_to_tracks` runs — so, within
  the live path, this is a **whole-track, full-lifetime, one-shot-per-run**
  decision, not a first-frame lock and not a per-frame classification.

## F. Confidence / unknown

**None exists.** There is no margin, distance-to-cluster-boundary, or
confidence score computed anywhere in this file. `team_id` always resolves
to `1` or `2` — even the "not enough samples" and "kmeans is None" fallback
paths hard-default to `team_id = 1` rather than `None`/unknown. Any
consumer (drawing, possession) can rely on `team` always being present and
being `1` or `2` for every entry in `tracks['players']` — confirmed by
`test_outfield_players_do_get_a_team`.

## G. Temporal behavior — can a track change team mid-video?

**No**, not within a single `assign_teams_to_tracks()` call: the per-frame
write-out loop (`for frame_idx, frame in enumerate(frames): ... team_id =
self.player_team_dict.get(player_id, 1)`) reads the SAME cached value for
every frame that `player_id` appears in — by construction, one player_id has
exactly one team for the whole processed clip. There is no mechanism to
revisit or split a track's team assignment partway through, and no signal
(e.g. a sudden color-mode change) is checked for. If a single tracked
`player_id` actually spans two different real players (an ID switch/mixed
tracklet — see `TRACK_CONTAMINATION_TRACK4.md`), today's implementation has
no way to represent or detect that: it will confidently emit exactly one
team label for the whole track regardless.

## Summary table

| question | answer |
|---|---|
| ROI | chest sub-region of the player's own bbox (15-50% h, 25-75% w) |
| color space | raw BGR pixel median, no HSV/Lab/histogram |
| clustering | KMeans k=2, fit on first ≤20 frames only, refit every `assign_teams_to_tracks()` call |
| aggregation | per-track median over the WHOLE track's clean/bright samples, one-shot per run |
| confidence/unknown | none — always emits 1 or 2 |
| mid-track team change | never possible in the current implementation |
| dead code | `get_player_team()` (first-lock-forever) is unreachable from production; `use_top_half` constructor flag is ignored |
