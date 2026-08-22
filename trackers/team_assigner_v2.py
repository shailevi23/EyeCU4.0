"""
Team Assignment V2 -- robust color tracklet (Candidate B from the frozen
post-freeze benchmark in experiments/post_freeze/team_assignment_v2/).

NOT adopted as the production default: on the frozen 46-track development
benchmark (two NON-TEST matches), the legacy TeamAssigner scored 46/46
(100%) while this implementation scored 24/46 (52%), including a complete
collapse (0/22) on the Bayern match, where player bounding boxes are too
small for this ROI's quality-rejection rule to ever find a usable
observation. See experiments/post_freeze/team_assignment_v2/FINAL_RESULTS.md
for the full comparison. Kept available behind
`team_assignment_backend="v2"` for rollback/experimentation only -- the
production default remains the legacy `TeamAssigner`.

Unlike the benchmark harness (which was frozen to always predict, for a fair
head-to-head comparison), this production version returns team=None when a
track never has a single usable observation, and callers must tolerate that:
this module simply does not write 'team'/'team_color' onto such a player's
tracking data, so existing `.get('team_color', default)`-style consumers
already handle it unchanged.

A team classifier -- this one or the legacy one -- can only ever affect
jersey-affiliation *presentation* for tracks that are, in fact, one
consistent identity. It cannot repair a genuine CBIoU ID switch (a track
whose underlying identity changed mid-life): see the MIXED_TRACK findings
in the same benchmark, where several tracks' color evidence is bimodal in a
way no team classifier should try to resolve into a single label.
"""
from typing import Dict, List

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

ROI = {'y_start': 0.15, 'y_end': 0.50, 'x_start': 0.25, 'x_end': 0.75}
MIN_PIXELS = 150
GREEN_HUE_RANGE = (35, 85)  # OpenCV 0-179 hue scale
GREEN_MIN_S = 60
GREEN_MIN_V = 60
MIN_NON_GREEN_FRACTION = 0.40
MAX_OBSERVATIONS_PER_TRACK = 9


class TeamAssignerV2:
    """Robust color tracklet team assigner -- see module docstring for status."""

    def __init__(self, num_teams: int = 2):
        self.num_teams = num_teams
        self.display_colors = {1: (255, 60, 60), 2: (60, 60, 255)}
        self.player_team_dict: Dict[int, int] = {}

    def _chest_roi(self, frame: np.ndarray, bbox: List[float]):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        ys, ye = int(h * ROI['y_start']), int(h * ROI['y_end'])
        xs, xe = int(w * ROI['x_start']), int(w * ROI['x_end'])
        torso = crop[ys:ye, xs:xe]
        return torso if torso.size else None

    def _descriptor_or_none(self, torso_bgr):
        if torso_bgr is None:
            return None
        n_pixels = torso_bgr.shape[0] * torso_bgr.shape[1]
        if n_pixels < MIN_PIXELS:
            return None

        hsv = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0].astype(np.float64), hsv[..., 1].astype(np.float64), hsv[..., 2].astype(np.float64)
        lo, hi = GREEN_HUE_RANGE
        is_green = (H >= lo) & (H <= hi) & (S >= GREEN_MIN_S) & (V >= GREEN_MIN_V)
        keep = ~is_green
        if keep.sum() / H.size < MIN_NON_GREEN_FRACTION:
            return None

        lab = cv2.cvtColor(torso_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
        L, a, b = lab[..., 0][keep], lab[..., 1][keep], lab[..., 2][keep]
        if L.size == 0:
            return None
        S_keep, H_keep = S[keep], H[keep]
        H_rad = H_keep * (np.pi / 90.0)
        return np.array([np.median(L), np.median(a), np.median(b),
                          np.median(S_keep), np.median(np.sin(H_rad)), np.median(np.cos(H_rad))])

    def assign_teams_to_tracks(self, frames: List[np.ndarray], tracks: Dict) -> None:
        n = len(tracks['players'])
        appearances: Dict[int, List[int]] = {}
        for idx in range(n):
            for tid in tracks['players'][idx].keys():
                appearances.setdefault(tid, []).append(idx)

        track_descriptors = {}
        for tid, idxs in appearances.items():
            usable = []
            sample_positions = np.linspace(0, len(idxs) - 1,
                                           min(len(idxs), MAX_OBSERVATIONS_PER_TRACK * 3)).round().astype(int)
            sample_positions = sorted(set(sample_positions.tolist()))
            for pos in sample_positions:
                if len(usable) >= MAX_OBSERVATIONS_PER_TRACK:
                    break
                idx = idxs[pos]
                frame = frames[idx] if idx < len(frames) else None
                if frame is None:
                    continue
                bbox = tracks['players'][idx][tid]['bbox']
                desc = self._descriptor_or_none(self._chest_roi(frame, bbox))
                if desc is not None:
                    usable.append(desc)
            if usable:
                track_descriptors[tid] = np.median(np.stack(usable, axis=0), axis=0)
            # else: no usable observation anywhere in this track's lifetime --
            # left out of track_descriptors entirely, so it gets no team
            # below (team=unknown), rather than a guessed label.

        tids = sorted(track_descriptors.keys())
        if len(tids) >= self.num_teams:
            X = np.stack([track_descriptors[tid] for tid in tids], axis=0)
            Xs = StandardScaler().fit_transform(X)
            km = KMeans(n_clusters=self.num_teams, n_init=20, random_state=0)
            cluster_labels = km.fit_predict(Xs)
            for tid, cl in zip(tids, cluster_labels):
                self.player_team_dict[tid] = int(cl) + 1
        # else: not enough tracks with any usable observation to even fit
        # k=2 this match -- every player stays unknown (no forced default).

        for frame_idx in range(n):
            for player_id, player_data in tracks['players'][frame_idx].items():
                team_id = self.player_team_dict.get(player_id)
                if team_id is None:
                    continue  # unknown: leave 'team'/'team_color' unset, not a guess
                player_data['team'] = team_id
                player_data['team_color'] = self.display_colors.get(team_id, (0, 255, 0))
