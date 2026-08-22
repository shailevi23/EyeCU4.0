"""
Candidate A -- exact current production TeamAssigner, unmodified, run on the
frozen 57-track benchmark's two matches. Zero detector inference: tracks
come from the existing caches, frames are plain video reads.
"""
import os
import sys
import time
import pickle
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
from trackers.team_assigner import TeamAssigner

HERE = os.path.dirname(os.path.abspath(__file__))

MATCHES = [
    {'match_id': 'bayern_munich_3-1_chelsea',
     'video': os.path.join('input-videos', 'Bayern Munich 3-1 Chelsea.mp4'),
     'cache': os.path.join('demo_outputs', 'final_e2e_demo', 'cache', 'tracks_b3bacf1707645184.pkl'),
     'skip_frames': 2},
    {'match_id': 'chelsea_v_leeds_united',
     'video': os.path.join('input-videos', 'Chelsea v Leeds United.mp4'),
     'cache': os.path.join('experiments', 'post_freeze', 'team_assignment_v2',
                           'match2_cache', 'cache', 'tracks_7562185f8dd2d6c6.pkl'),
     'skip_frames': 2},
]


def read_frames(video_path, n_processed, skip_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for i in range(n_processed):
        raw_idx = i * skip_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, raw_idx)
        ok, frame = cap.read()
        frames.append(frame if ok else None)
    cap.release()
    return frames


def main():
    predictions = {}
    runtimes = {}
    for match in MATCHES:
        with open(match['cache'], 'rb') as f:
            blob = pickle.load(f)
        tracks = blob['tracks']
        n = len(tracks['players'])
        frames = read_frames(match['video'], n, match['skip_frames'])

        ta = TeamAssigner(num_teams=2)
        t0 = time.time()
        ta.assign_teams_to_tracks(frames, tracks)
        runtime_s = time.time() - t0
        runtimes[match['match_id']] = runtime_s

        match_preds = {}
        for tid, team_id in ta.player_team_dict.items():
            match_preds[tid] = int(team_id)
        predictions[match['match_id']] = match_preds
        print(f"[{match['match_id']}] assign_teams_to_tracks runtime: {runtime_s:.3f}s, "
              f"{len(match_preds)} unique player track ids assigned")

    out = {'predictions': predictions, 'runtimes': runtimes}
    out_path = os.path.join(HERE, 'candidate_A_predictions.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == '__main__':
    main()
