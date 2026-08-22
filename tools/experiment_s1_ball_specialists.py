#!/usr/bin/env python
"""
S1 -- non-YOLO soccer ball specialist gate. DIAGNOSTIC / EXPERIMENT ONLY.

Runs the official WASB-SBDT Soccer-pretrained specialists (DeepBall,
DeepBall-Large, BallSeg, TrackNetV2, ResTrackNetV2) on the frozen EyeCU temporal
benchmark and scores them against the frozen R1 populations with the OFFICIAL
localization rule from that framework's own evaluator.

Nothing about EyeCU is changed. No training, no fine-tuning, no SAHI, no TTA, no
BallTemporalSelector, no interpolation, no annotation, no threshold search.
Specialist architectures are given the adjacent source-video frames their
official input protocol requires -- that is part of their architecture, and it is
raw architecture output that is scored.

Official rule, lifted from WASB-SBDT src/utils/evaluator.py and
src/configs/runner/eval.yaml (NOT invented here):

    heatmap/probability accept threshold  0.5
    per-frame selection                   tracker=intra_frame_peak (max score)
    localization                           L2(pred_centre, gt_centre) < 4 px,
                                           in ORIGINAL frame coordinates
    TP   gt visible, predicted, dist < 4
    FP1  gt visible, predicted, dist >= 4
    FP2  gt not visible, predicted
    FN   gt visible, not predicted
    TN   gt not visible, not predicted

YOLO baselines are scored by the identical rule using bbox centre, at the frozen
production accept threshold 0.25 after the frozen ball dedupe.

VALIDATION ONLY. The benchmark is VAL_ONLY by construction; sealed TEST is
unreachable from this file.

    python tools/experiment_s1_ball_specialists.py --wasb-root <dir> --out-dir <dir>
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trackers.detector import (BALL_ACCEPT_CONF, BALL_CANDIDATE_CONF,   # noqa: E402
                               BALL_DEDUPE_IOU, LocalDetector,
                               suppress_ball_duplicates)

TV = Path('data/temporal_val')
R1 = Path('EyeCU_R1_results/r1_20260817T144649Z/R1_RESULTS.json')

# ---- official constants, copied from the WASB-SBDT configs -----------------
OFFICIAL_SCORE_THRESHOLD = 0.5     # configs/runner/eval.yaml + detector/*.yaml
OFFICIAL_DIST_THRESHOLD = 4.0      # configs/runner/eval.yaml: dist_threshold: 4
# Predeclared BEFORE any candidate was run: the official 4 px is native to the
# WASB Soccer dataset (ISSIA-CNR, 1920x1080). Our frames are 640x360, so the
# width-normalised equivalent is 4 * 640/1920. Both are reported; the literal
# official 4 px is PRIMARY, per the instruction to use the official rule exactly.
NORMALISED_DIST_THRESHOLD = 4.0 * 640.0 / 1920.0
WASB_SOCCER_NATIVE_WIDTH = 1920

# YOLO baselines, frozen EyeCU production thresholds.
YOLO_CELLS = [('A@960', 'best_A_960.pt', 960),
              ('B@1280', 'best_B_1280.pt', 1280),
              ('B@1920', 'best_B_1920_is_B_at_1920', 1920)]
YOLO_WEIGHTS = {'A@960': 'best_A_960.pt', 'B@1280': 'best_B_1280.pt',
                'B@1920': 'best_B_1280.pt'}

# Candidate table. Every field is taken from the official configs; nothing tuned.
CANDIDATES = {
    'DeepBall': {
        'ckpt': 'deepball_soccer_best.pth.tar',
        'model': {'name': 'deepball', 'frames_in': 1, 'frames_out': 1,
                  'out_scales': [0], 'class_out': 2, 'foreground_channel': 1,
                  'rgb_diff': False, 'inp_height': 720, 'inp_width': 1280,
                  'out_height': 180, 'out_width': 320,
                  'block_channels': [8, 16, 32],
                  'block_maxpools': [True, True, True],
                  'first_conv_kernel_size': 7, 'first_conv_stride': 2,
                  'last_conv_kernel_size': 3},
        'pp': 'deepball', 'output_kind': 'softmax peak (point)'},
    'DeepBall-Large': {
        'ckpt': 'deepball-large_soccer_best.pth.tar',
        'model': {'name': 'deepball', 'frames_in': 1, 'frames_out': 1,
                  'out_scales': [0], 'class_out': 2, 'foreground_channel': 1,
                  'rgb_diff': False, 'inp_height': 720, 'inp_width': 1280,
                  'out_height': 180, 'out_width': 320,
                  'block_channels': [32, 64, 128],
                  'block_maxpools': [True, True, True],
                  'first_conv_kernel_size': 7, 'first_conv_stride': 2,
                  'last_conv_kernel_size': 3},
        'pp': 'deepball', 'output_kind': 'softmax peak (point)'},
    'BallSeg': {
        'ckpt': 'ballseg_soccer_best.pth.tar',
        'model': {'name': 'ballseg', 'frames_in': 2, 'frames_out': 1,
                  'out_scales': [0], 'rgb_diff': True,
                  'inp_height': 576, 'inp_width': 1024,
                  'out_height': 576, 'out_width': 1024,
                  'backbone': 'resnet18', 'scale_factors': [1, 1, 0.5]},
        'pp': 'tracknetv2', 'output_kind': 'segmentation blob'},
    'TrackNetV2': {
        'ckpt': 'tracknetv2_soccer_best.pth.tar',
        'model': {'name': 'tracknetv2', 'frames_in': 3, 'frames_out': 3,
                  'inp_height': 288, 'inp_width': 512,
                  'out_height': 288, 'out_width': 512, 'bilinear': True,
                  'halve_channel': False, 'mode': 'nearest', 'rgb_diff': False,
                  'out_scales': [0]},
        'pp': 'tracknetv2', 'output_kind': 'heatmap blob'},
    'ResTrackNetV2': {
        'ckpt': 'restracknetv2_soccer_best.pth.tar',
        'model': {'name': 'restracknetv2', 'frames_in': 3, 'frames_out': 3,
                  'inp_height': 288, 'inp_width': 512,
                  'out_height': 288, 'out_width': 512, 'rgb_diff': False,
                  'out_scales': [0], 'halve_channel': False, 'mode': 'nearest',
                  'out_mid_channels': 64, 'neck_channels': 64,
                  'blocks': [3, 3, 4, 3], 'channels': [16, 32, 64, 128]},
        'pp': 'tracknetv2', 'output_kind': 'heatmap blob'},
}

# For frames_in == frames_out == 3 every output index is officially valid (the
# eval runner maps output j to input frame j and step=3 tiles the clip, so a
# frame is scored at whichever position it falls in). We predeclare CENTRE as
# primary and report all three positions as a robustness check.
PRIMARY_POSITION = 1


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_official_modules(wasb_src: Path):
    """
    Import the official inference modules we need WITHOUT executing
    utils/__init__.py or dataloaders/__init__.py, which pull in hydra/omegaconf.
    Those are the framework's CLI/config layer; no inference code path touches
    them. The model definitions, the postprocessors and the coordinate maths are
    the official files, loaded unmodified.
    """
    import importlib.util
    import types

    for pkg in ('utils', 'detectors'):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(wasb_src / pkg)]
            sys.modules[pkg] = m

    def load(modname, relpath):
        if modname in sys.modules:
            return sys.modules[modname]
        spec = importlib.util.spec_from_file_location(modname, wasb_src / relpath)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        return mod

    load('utils.image', 'utils/image.py')
    uu = load('utils.utils', 'utils/utils.py')
    sys.modules['utils'].read_image = uu.read_image
    load('detectors.postprocessor', 'detectors/postprocessor.py')
    load('detectors.deepball_postprocessor', 'detectors/deepball_postprocessor.py')


def get_transform(img, input_wh, inv=0):
    """Verbatim copy of WASB-SBDT src/dataloaders/dataset_loader.py:20-26.
    Reproduced here only to avoid importing that module's omegaconf-dependent
    package __init__; the maths is unchanged."""
    from utils.image import get_affine_transform                    # noqa: PLC0415
    h, w, _ = img.shape
    c = np.array([w / 2., h / 2.], dtype=np.float32)
    s = max(h, w) * 1.0
    input_w, input_h = input_wh
    return get_affine_transform(c, s, 0, [input_w, input_h], inv=inv)


def load_gt_centre(stem: str, w: int, h: int):
    """Frozen benchmark labels are single-class (0 = ball). Returns list of (cx, cy)."""
    p = TV / 'labels' / f'{stem}.txt'
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        q = line.split()
        if len(q) == 5:
            out.append((float(q[1]) * w, float(q[2]) * h))
    return out


# ------------------------------------------------------------------ frames

class SourceFrames:
    """Contiguous original-video frames, keyed by absolute source frame index."""

    def __init__(self):
        import cv2
        self.cv2 = cv2
        self._caps = {}
        # Only ~520 distinct 640x360 frames are ever needed (5 per target), so
        # cache them all rather than re-seeking the container hundreds of times.
        self._cache = {}

    def cap(self, video: str):
        if video not in self._caps:
            c = self.cv2.VideoCapture(video)
            if not c.isOpened():
                raise RuntimeError(f'cannot open {video}')
            self._caps[video] = c
        return self._caps[video]

    def get(self, video: str, index: int):
        key = (video, index)
        if key in self._cache:
            return self._cache[key]
        c = self.cap(video)
        c.set(self.cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = c.read()
        self._cache[key] = frame if ok else None
        return self._cache[key]

    def close(self):
        for c in self._caps.values():
            c.release()


def validate_alignment(frames_meta, src, radius=2):
    """
    Scripted proof of frame alignment. For every target the frozen benchmark JPEG
    must be closest (mean absolute pixel difference) to the source frame at the
    manifest's source_frame_index, and to no neighbour within +/-radius.
    """
    import cv2
    rows, bad = [], []
    for f in frames_meta:
        frozen = cv2.imdecode(np.fromfile(str(TV / 'images' / f['file']),
                                         dtype=np.uint8), cv2.IMREAD_COLOR)
        fi = f['source_frame_index']
        mads = {}
        for d in range(-radius, radius + 1):
            g = src.get(f['source_video'], fi + d)
            if g is None or g.shape != frozen.shape:
                continue
            mads[d] = float(np.abs(g.astype(np.int16)
                                   - frozen.astype(np.int16)).mean())
        best = min(mads, key=mads.get) if mads else None
        ok = best == 0
        rows.append({'file': f['file'], 'source_frame_index': fi,
                     'shape': list(frozen.shape[:2]), 'argmin_offset': best,
                     'mad_at_0': round(mads.get(0, -1), 4),
                     'mad_next_best': round(min((v for k, v in mads.items()
                                                 if k != 0), default=-1), 4),
                     'aligned': ok})
        if not ok:
            bad.append(f['file'])
    return rows, bad


# ------------------------------------------------------------------ WASB

def build_wasb(name, spec, wasb_root: Path, weights_dir: Path):
    """Load one official checkpoint on CPU, bypassing only the framework's
    hard `device=cuda` assert in detectors/detector.py (no GPU on this host)."""
    import torch
    src = wasb_root / 'src'
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    load_official_modules(src)
    from models import build_model                                  # noqa: PLC0415

    ck_path = weights_dir / spec['ckpt']
    cfg = {'model': spec['model'],
           'detector': {'postprocessor': {
               'name': spec['pp'],
               'score_threshold': OFFICIAL_SCORE_THRESHOLD,
               'scales': [0], 'blob_det_method': 'concomp',
               'use_hm_weight': False}},
           'dataloader': {'heatmap': {'sigmas': [2.5]}}}
    model = build_model(cfg)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    sd = ck['model_state_dict']
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    model.load_state_dict(sd)          # strict: proves architecture identity
    model.eval()

    if spec['pp'] == 'deepball':
        from detectors.deepball_postprocessor import DeepBallPostprocessor as PP
    else:
        from detectors.postprocessor import TracknetV2Postprocessor as PP
    return model, PP(cfg), cfg, ck_path, sum(p.numel() for p in model.parameters())


def wasb_predict(model, pp, spec, imgs_bgr):
    """One official forward pass. imgs_bgr in chronological order, len==frames_in.
    Returns {out_index: [{'xy': (x, y) in ORIGINAL coords, 'score': s}]}."""
    import cv2
    import torch
    from PIL import Image
    from torchvision import transforms as T

    m = spec['model']
    inp_wh = (m['inp_width'], m['inp_height'])
    out_wh = (m['out_width'], m['out_height'])
    norm = T.Compose([T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

    ref = imgs_bgr[0]
    trans_in = get_transform(ref, inp_wh)
    trans_out_inv = get_transform(ref, out_wh, inv=1)

    tens = []
    for bgr in imgs_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        warped = cv2.warpAffine(rgb, trans_in, inp_wh, flags=cv2.INTER_LINEAR)
        tens.append(norm(Image.fromarray(warped)))
    if m['rgb_diff']:
        tens[0] = torch.abs(tens[1] - tens[0])
    x = torch.cat(tens, dim=0).unsqueeze(0)

    with torch.no_grad():
        preds = model(x)
    affine = {0: torch.from_numpy(trans_out_inv).unsqueeze(0).float()}
    res = pp.run(preds, affine)

    out = {}
    for eid in sorted(res[0].keys()):
        dets = []
        for xy, sc in zip(res[0][eid][0]['xys'], res[0][eid][0]['scores']):
            dets.append({'xy': (float(xy[0]), float(xy[1])), 'score': float(sc)})
        out[eid] = dets
    return out


def wasb_raw_heatmap(model, spec, imgs_bgr, out_index):
    """
    Raw activation maximum for one frame, BEFORE the official 0.5 accept
    threshold. Uses the same official activation as each postprocessor
    (softmax-foreground for DeepBall, sigmoid for the heatmap models), so it
    shows whether the accept threshold is the binding constraint.
    DIAGNOSTIC ONLY -- no threshold is changed anywhere.
    """
    import cv2
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision import transforms as T

    m = spec['model']
    inp_wh = (m['inp_width'], m['inp_height'])
    norm = T.Compose([T.ToTensor(),
                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    trans_in = get_transform(imgs_bgr[0], inp_wh)
    tens = []
    for bgr in imgs_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        warped = cv2.warpAffine(rgb, trans_in, inp_wh, flags=cv2.INTER_LINEAR)
        tens.append(norm(Image.fromarray(warped)))
    if m['rgb_diff']:
        tens[0] = torch.abs(tens[1] - tens[0])
    x = torch.cat(tens, dim=0).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)[0]
    if spec['pp'] == 'deepball':
        hm = F.softmax(logits, dim=1)[:, m['foreground_channel']]
    else:
        hm = torch.sigmoid(logits)[:, out_index]
    return float(hm.max())


def intra_frame_peak(dets):
    """Official tracker=intra_frame_peak: highest-scoring detection, or none."""
    if not dets:
        return None
    return max(dets, key=lambda d: d['score'])


# ------------------------------------------------------------------ scoring

def score(predictions, frames_meta, gt_centres, hard, thresh):
    """
    predictions: file -> {'xy': (x, y), 'score': s} or None.
    Returns the official confusion counts plus the frozen hard-set breakdown.
    """
    tp = fp1 = fp2 = fn = tn = 0
    empty_fired = 0
    hit_files = set()
    for f in frames_meta:
        g = gt_centres[f['file']]
        p = predictions.get(f['file'])
        if g:
            if p is None:
                fn += 1
            else:
                d = min(float(np.hypot(p['xy'][0] - c[0], p['xy'][1] - c[1]))
                        for c in g)
                if d < thresh:
                    tp += 1
                    hit_files.add(f['file'])
                else:
                    fp1 += 1
        else:
            if p is None:
                tn += 1
            else:
                fp2 += 1
                empty_fired += 1

    ev = defaultdict(int)
    rec = ovl = 0
    for m in hard:
        if m['file'] in hit_files:
            rec += 1
            ev[m['event']] += 1
            ovl += bool(m['human_overlap'])
    n_gt = tp + fp1 + fn
    return {
        'tp': tp, 'fp1': fp1, 'fp2': fp2, 'fn': fn, 'tn': tn,
        'gt_frames': n_gt,
        'recall': round(tp / n_gt, 4) if n_gt else None,
        'precision': (round(tp / (tp + fp1 + fp2), 4)
                      if (tp + fp1 + fp2) else None),
        'empty_frames': tp * 0 + (fp2 + tn),
        'empty_fired_frames': empty_fired,
        'empty_fired_rate': round(empty_fired / (fp2 + tn), 4) if (fp2 + tn) else None,
        'hard_recovered': rec, 'hard_of': len(hard),
        'overlap_recovered': ovl,
        'overlap_of': sum(bool(m['human_overlap']) for m in hard),
        'events_touched': len(ev),
        'events_of': len({m['event'] for m in hard}),
        'events_with_2plus': sum(1 for v in ev.values() if v >= 2),
        'hard_recovered_files': sorted(hit_files & {m['file'] for m in hard}),
    }


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--wasb-root', required=True)
    ap.add_argument('--weights-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--skip-yolo', action='store_true')
    args = ap.parse_args()

    import cv2
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    wasb_root = Path(args.wasb_root)
    weights_dir = Path(args.weights_dir)

    man = json.loads((TV / 'manifest.json').read_text(encoding='utf-8'))
    frames_meta = man['frames']
    by_window = defaultdict(list)
    for f in frames_meta:
        by_window[(f['match'], f['window'])].append(f)
    for v in by_window.values():
        v.sort(key=lambda f: f['order_in_window'])

    # frozen geometry + GT centres
    gt_centres, sizes = {}, {}
    for f in frames_meta:
        img = cv2.imdecode(np.fromfile(str(TV / 'images' / f['file']),
                                       dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        sizes[f['file']] = (w, h)
        gt_centres[f['file']] = load_gt_centre(Path(f['file']).stem, w, h)

    n_gt_frames = sum(1 for f in frames_meta if gt_centres[f['file']])
    n_gt_objs = sum(len(gt_centres[f['file']]) for f in frames_meta)
    n_empty = len(frames_meta) - n_gt_frames

    # frozen R1 hard set, loaded not redefined
    r1 = json.loads(R1.read_text(encoding='utf-8'))
    hard = r1['contact_set']['members']

    report = {
        'EXPERIMENT': 'S1 non-YOLO soccer ball specialist gate',
        'DIAGNOSTIC_ONLY': True,
        'device': 'cpu',
        'official_rule': {
            'source': 'WASB-SBDT src/utils/evaluator.py + configs/runner/eval.yaml',
            'score_threshold': OFFICIAL_SCORE_THRESHOLD,
            'dist_threshold_px_literal': OFFICIAL_DIST_THRESHOLD,
            'dist_threshold_px_width_normalised': round(NORMALISED_DIST_THRESHOLD, 4),
            'wasb_soccer_native_width': WASB_SOCCER_NATIVE_WIDTH,
            'eyecu_frame_width': 640,
            'coordinate_system': 'ORIGINAL frozen benchmark frame, 640x360',
            'selection': 'tracker=intra_frame_peak (max score, single ball)',
            'yolo_threshold': BALL_ACCEPT_CONF,
            'yolo_centre': 'predicted bbox centre'},
        'benchmark': {'targets': len(frames_meta), 'gt_ball_frames': n_gt_frames,
                      'gt_ball_objects': n_gt_objs, 'empty_frames': n_empty},
        'hard_set': {'n': len(hard),
                     'n_overlap': sum(bool(m['human_overlap']) for m in hard),
                     'n_events': len({m['event'] for m in hard}),
                     'distinct_files': len({m['file'] for m in hard})},
        'alignment': {}, 'candidates': {}, 'baselines': {},
    }

    # ---------------------------------------------------- frame alignment
    src = SourceFrames()
    rows, bad = validate_alignment(frames_meta, src)
    report['alignment'] = {
        'rule': ('frozen benchmark JPEG must be closest by mean-absolute pixel '
                 'difference to source_frame_index, and to no neighbour +/-2'),
        'targets_checked': len(rows), 'targets_aligned': len(rows) - len(bad),
        'misaligned': bad,
        'mad_at_0_max': round(max(r['mad_at_0'] for r in rows), 4),
        'mad_at_0_mean': round(float(np.mean([r['mad_at_0'] for r in rows])), 4),
        'mad_next_best_min': round(min(r['mad_next_best'] for r in rows), 4),
        'per_target': rows}
    print(f'alignment: {len(rows) - len(bad)}/{len(rows)} aligned, '
          f'MAD@0 max {report["alignment"]["mad_at_0_max"]}, '
          f'next-best min {report["alignment"]["mad_next_best_min"]}', flush=True)
    if bad:
        report['ABORT'] = 'frame alignment could not be proven'
        (out / 'S1_RESULTS.json').write_text(json.dumps(report, indent=1),
                                            encoding='utf-8')
        print('ABORT: alignment failed', flush=True)
        return 1

    # ---------------------------------------------------- YOLO baselines
    if not args.skip_yolo:
        for tag in ('A@960', 'B@1280', 'B@1920'):
            imgsz = int(tag.split('@')[1])
            det = LocalDetector(YOLO_WEIGHTS[tag], confidence=BALL_CANDIDATE_CONF,
                                iou=0.5, imgsz=imgsz, device='cpu',
                                ball_candidate_pool=False)
            preds, t0 = {}, time.time()
            for f in frames_meta:
                img = cv2.imdecode(np.fromfile(str(TV / 'images' / f['file']),
                                               dtype=np.uint8), cv2.IMREAD_COLOR)
                balls = [d for d in det.detect(img) if d['class'] == 'ball']
                kept = suppress_ball_duplicates(
                    [b for b in balls if b['confidence'] >= BALL_CANDIDATE_CONF],
                    BALL_DEDUPE_IOU)
                acc = [b for b in kept if b['confidence'] >= BALL_ACCEPT_CONF]
                best = max(acc, key=lambda d: d['confidence']) if acc else None
                preds[f['file']] = (None if best is None else {
                    'xy': ((best['bbox'][0] + best['bbox'][2]) / 2,
                           (best['bbox'][1] + best['bbox'][3]) / 2),
                    'score': best['confidence']})
            ms = (time.time() - t0) * 1000 / len(frames_meta)
            report['baselines'][tag] = {
                'family': 'YOLO26s (EyeCU)', 'frames_per_inference': 1,
                'single_or_multi': 'MULTI-BALL CAPABLE (scored single here)',
                'ms_per_target_cpu': round(ms, 1),
                'official_4px': score(preds, frames_meta, gt_centres, hard,
                                      OFFICIAL_DIST_THRESHOLD),
                'normalised_1p33px': score(preds, frames_meta, gt_centres, hard,
                                           NORMALISED_DIST_THRESHOLD)}
            s = report['baselines'][tag]['official_4px']
            print(f'{tag}: R {s["recall"]} hard {s["hard_recovered"]}/{s["hard_of"]} '
                  f'ovl {s["overlap_recovered"]}/{s["overlap_of"]} '
                  f'evt {s["events_touched"]}/{s["events_of"]} '
                  f'emptyfired {s["empty_fired_frames"]}', flush=True)

    # ---------------------------------------------------- candidates
    for name, spec in CANDIDATES.items():
        entry = {'family': spec['model']['name'],
                 'frames_in': spec['model']['frames_in'],
                 'frames_out': spec['model']['frames_out'],
                 'input_hw': [spec['model']['inp_height'], spec['model']['inp_width']],
                 'output_kind': spec['output_kind'],
                 'official_score_threshold': OFFICIAL_SCORE_THRESHOLD,
                 'license': 'MIT (WASB-SBDT LICENSE.md)',
                 'single_or_multi': 'SINGLE-BALL ONLY',
                 'ckpt_file': spec['ckpt']}
        ck = weights_dir / spec['ckpt']
        if not ck.exists():
            entry['evaluable'] = 'NOT EVALUABLE -- official weights not obtainable'
            report['candidates'][name] = entry
            print(f'{name}: NOT EVALUABLE (no weights)', flush=True)
            continue
        try:
            model, pp, cfg, ck_path, nparam = build_wasb(name, spec, wasb_root,
                                                         weights_dir)
        except Exception as e:                                     # noqa: BLE001
            entry['evaluable'] = f'NOT EVALUABLE -- {type(e).__name__}: {e}'
            report['candidates'][name] = entry
            print(f'{name}: NOT EVALUABLE ({type(e).__name__}: {e})', flush=True)
            continue

        entry.update({'evaluable': 'YES', 'params': nparam,
                      'ckpt_bytes': ck.stat().st_size,
                      'ckpt_sha256': sha256(ck)})

        F, O = spec['model']['frames_in'], spec['model']['frames_out']
        positions = [PRIMARY_POSITION] if (F == 3 and O == 3) else [0]
        all_pos = [0, 1, 2] if (F == 3 and O == 3) else [0]
        per_pos = {}
        t0 = time.time()
        n_fwd = 0
        for pos in all_pos:
            preds = {}
            for f in frames_meta:
                fi = f['source_frame_index']
                idxs = [fi - pos + k for k in range(F)]
                imgs = [src.get(f['source_video'], i) for i in idxs]
                if any(i is None for i in imgs):
                    preds[f['file']] = None
                    continue
                res = wasb_predict(model, pp, spec, imgs)
                n_fwd += 1
                preds[f['file']] = intra_frame_peak(res.get(pos, []))
            per_pos[f'pos{pos}'] = {
                'official_4px': score(preds, frames_meta, gt_centres, hard,
                                      OFFICIAL_DIST_THRESHOLD),
                'normalised_1p33px': score(preds, frames_meta, gt_centres, hard,
                                           NORMALISED_DIST_THRESHOLD)}
            s = per_pos[f'pos{pos}']['official_4px']
            print(f'{name} pos{pos}: R {s["recall"]} '
                  f'hard {s["hard_recovered"]}/{s["hard_of"]} '
                  f'ovl {s["overlap_recovered"]}/{s["overlap_of"]} '
                  f'evt {s["events_touched"]}/{s["events_of"]} '
                  f'emptyfired {s["empty_fired_frames"]}', flush=True)
        entry['ms_per_target_cpu'] = round((time.time() - t0) * 1000 / max(n_fwd, 1), 1)
        entry['positions'] = per_pos
        entry['primary_position'] = f'pos{positions[0]}'
        entry['primary'] = per_pos[f'pos{positions[0]}']
        report['candidates'][name] = entry

    src.close()
    (out / 'S1_RESULTS.json').write_text(json.dumps(report, indent=1),
                                        encoding='utf-8')
    print(f'\nwritten: {out / "S1_RESULTS.json"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
