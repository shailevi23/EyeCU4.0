#!/usr/bin/env python
"""
Build MP4 clips from the frozen package frames, for CVAT VIDEO tasks.

CVAT decides interpolation vs annotation mode from what you upload:

    video  -> interpolation mode -> TRACK objects   <- what this benchmark needs
    images -> annotation mode    -> SHAPE objects

Our importer reads `<track>` semantics, so the task must be created from a
video. Uploading the `img1/` JPEG folder produces an image task, and no amount
of care during annotation recovers identity from it.

The MP4 is a UI VEHICLE ONLY. Canonical provenance stays with the frozen
package: `img1/000001.jpg` is package frame 1, which is source frame
`source_frame_range[0]`. The clip exists so CVAT will offer track mode, and its
decoded frame k (0-based) must correspond to package frame k+1 -- nothing more
is asked of it, and nothing measured is derived from its pixels.

Because the whole point is that mapping, every clip is verified after encoding
by decoding it back:

    exact frame count           decoded count == frame_count
    frame 0 -> package frame 1  decoded frame 0 best-matches 000001.jpg
    ordered and complete        every decoded frame matches ITS OWN package
                                frame better than any frame within +/-2
    no shift                    offset 0 minimises total difference over the
                                whole clip AND over each third, so a shift
                                starting midway cannot hide
    no duplicates               no two consecutive decoded frames identical
                                where the package frames differ
    geometry                    640x360 preserved
    frame rate                  container fps == the sequence's native fps

Alignment is decided by argmin, not by a ratio. Football footage contains
stretches where consecutive frames differ less than the encoder's own noise --
a wide static shot before a restart, for instance -- and there a
"beat your neighbour by 1.5x" rule fails on perfectly aligned frames simply
because there is nothing to distinguish. argmin is scale-free and does not
care. The ratio is still computed and reported wherever the neighbouring
package frames are distinguishable at all, as a strength-of-evidence figure.

The verification record is written next to the clip. A clip that fails is
deleted rather than shipped: a silently misaligned video would put every
annotation on the wrong frame.

Encoding uses ffmpeg with a constant frame rate and no frame-rate filtering.
`-vsync passthrough` (aka `-fps_mode passthrough`) is what stops ffmpeg from
duplicating or dropping frames to hit a timing target, which is exactly the
failure this tool must not have.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CRF = '12'                 # visually lossless enough for annotation
NEIGHBOUR_MARGIN = 1.5     # own frame must beat neighbours by this factor


def imread(p: Path):
    if not p.exists():
        return None
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


def mad(a, b):
    if a is None or b is None or a.shape != b.shape:
        return float('inf')
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


def fps_fraction(fps: float) -> str:
    """Fallback rational for a float fps, when ffprobe cannot be asked."""
    f = Fraction(fps).limit_denominator(1000)
    return f'{f.numerator}/{f.denominator}'


def exact_fps(source_video: str, declared: float) -> tuple[str, str]:
    """
    (rational string, how it was obtained).

    The manifest stores what OpenCV reported, which for NTSC sources is the
    rounded 29.97 rather than the true 30000/1001. ffprobe reports the exact
    rational the container carries, so ask it rather than reconstruct it, and
    refuse a value that disagrees with the manifest.
    """
    try:
        p = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of',
             'default=nw=1:nk=1', source_video],
            capture_output=True, text=True, timeout=60)
        r = p.stdout.strip()
        if p.returncode == 0 and '/' in r:
            val = float(Fraction(r))
            if abs(val - declared) <= 0.02:
                return r, 'ffprobe r_frame_rate of the source video'
            raise SystemExit(
                f'{source_video}: ffprobe says {r} ({val:.5f} fps) but the '
                f'manifest records native_fps {declared}. Refusing to guess.')
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return fps_fraction(declared), 'derived from manifest native_fps'


def encode(img1: Path, out: Path, n: int, r: str, start: int = 1):
    """Encode frames start..start+n-1 of img1 into out, in order, at rate r."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
           '-framerate', r,
           '-start_number', str(start),
           '-i', str(img1 / '%06d.jpg'),
           '-frames:v', str(n),
           '-c:v', 'libx264', '-preset', 'slow', '-crf', CRF,
           '-pix_fmt', 'yuv420p',
           # passthrough keeps one output frame per input image; ffmpeg 8
           # rejects an explicit -r alongside it, and none is needed because
           # -framerate already fixed the rate on a CFR image sequence
           '-fps_mode', 'passthrough',
           '-movflags', '+faststart',
           str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f'ffmpeg failed:\n{proc.stderr[:2000]}')


def verify(clip: Path, img1: Path, n: int, fps: float, W: int, H: int,
           start: int = 1):
    """Decode the clip back and check it really is those frames, in order."""
    checks, ok_all = [], True

    def note(name, ok, detail=''):
        nonlocal ok_all
        ok_all &= bool(ok)
        checks.append({'check': name, 'ok': bool(ok), 'detail': detail})

    cap = cv2.VideoCapture(str(clip))
    note('clip opens', cap.isOpened())
    got_fps = cap.get(cv2.CAP_PROP_FPS)
    note('container fps matches native fps', abs(got_fps - fps) < 0.02,
         f'{got_fps:.5f} vs {fps:.5f}')
    note('width x height preserved',
         int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == W
         and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == H,
         f'{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x'
         f'{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')

    decoded = []
    while True:
        got, fr = cap.read()
        if not got:
            break
        decoded.append(fr)
    cap.release()
    note('exact frame count', len(decoded) == n, f'{len(decoded)} decoded, {n} expected')

    pkg = [imread(img1 / f'{start + k:06d}.jpg') for k in range(-1, n + 1)]

    def pkg_at(k):
        """Package frame for decoded index k, or None outside the clip."""
        return pkg[k + 1] if -1 <= k <= n else None

    mapping, worst_ratio, misaligned, dupes = [], float('inf'), [], []
    low_margin, unresolvable = [], 0
    for k, fr in enumerate(decoded[:n]):
        pf = start + k
        own = mad(fr, pkg_at(k))

        # Which nearby package frame does this decoded frame actually match?
        # argmin is the right test: it is scale-free, so it works whether the
        # scene is moving or static.
        cands = [(mad(fr, pkg_at(k + o)), o) for o in (-2, -1, 1, 2)
                 if pkg_at(k + o) is not None]
        best_other, best_off = min(cands) if cands else (float('inf'), None)
        ratio = best_other / own if own > 0 else float('inf')

        # A margin only means something when the neighbouring PACKAGE frames
        # differ from each other by more than the encoding noise. Where they do
        # not, no pixel comparison can separate them -- that is a fact about the
        # footage, not a failure. Those frames are counted, not silently passed.
        steps = [mad(pkg_at(k), pkg_at(k + o)) for o in (-1, 1)
                 if pkg_at(k + o) is not None]
        resolvable = bool(steps) and max(steps) > own

        if best_off is not None and best_other <= own:
            misaligned.append({'decoded_frame': k, 'package_frame': pf,
                               'mad': round(own, 3),
                               'better_match_at_offset': best_off,
                               'its_mad': round(best_other, 3)})
        if resolvable:
            worst_ratio = min(worst_ratio, ratio)
            if ratio <= NEIGHBOUR_MARGIN:
                low_margin.append({'decoded_frame': k, 'package_frame': pf,
                                   'mad': round(own, 3),
                                   'neighbour_ratio': round(ratio, 3)})
        else:
            unresolvable += 1

        if k > 0 and mad(fr, decoded[k - 1]) == 0.0 and mad(pkg_at(k), pkg_at(k - 1)) > 0.0:
            dupes.append(k)
        mapping.append({'decoded_frame_0based': k, 'package_frame_1based': pf,
                        'source_frame': None, 'mad': round(own, 3)})

    # A shift that starts midway would leave most frames looking fine, so check
    # each third as a whole as well as the clip as a whole.
    shifts = {}
    m = min(n, len(decoded))      # a short clip already failed the count check
    for name, lo, hi in (('all', 0, m), ('first_third', 0, m // 3),
                         ('middle_third', m // 3, 2 * m // 3),
                         ('last_third', 2 * m // 3, m)):
        tots = []
        for off in (-2, -1, 0, 1, 2):
            v = [mad(decoded[k], pkg_at(k + off)) for k in range(lo, hi)
                 if pkg_at(k + off) is not None]
            tots.append(sum(v) / len(v) if v else float('inf'))
        shifts[name] = {'by_offset': [round(t, 4) for t in tots],
                        'argmin_offset': (-2, -1, 0, 1, 2)[tots.index(min(tots))]}

    note('decoded frame 0 is package frame 1',
         bool(decoded) and mapping and mapping[0]['package_frame_1based'] == 1
         and not any(m['decoded_frame'] == 0 for m in misaligned))
    note('every decoded frame matches its own package frame', not misaligned,
         f'{len(misaligned)} misaligned' if misaligned else
         f'{n}/{n} by argmin over offsets -2..+2')
    note('no whole-clip or per-segment shift',
         all(v['argmin_offset'] == 0 for v in shifts.values()),
         ', '.join(f'{k}:{v["argmin_offset"]:+d}' for k, v in shifts.items()))
    note('no duplicated frames', not dupes, str(dupes[:5]))
    # Reported, never a gate. A margin is only meaningful where consecutive
    # frames differ; where they do not, a low ratio says the footage is static,
    # not that the mapping is wrong -- and annotating on a frame identical to
    # its neighbour has no consequence for the GT either way.
    checks.append({'check': 'neighbour-margin statistics (informational)',
                   'ok': True,
                   'detail': f'{n - unresolvable}/{n} frames have neighbours '
                             f'distinguishable above encode noise; weakest '
                             f'margin {worst_ratio:.2f}x; '
                             f'{len(low_margin)} below {NEIGHBOUR_MARGIN}x'})
    return (ok_all, checks, mapping, worst_ratio, shifts, unresolvable,
            misaligned, low_margin)


def build_one(root: Path, s: dict, out_dir: Path, n: int, tag: str):
    seq = s['sequence']
    img1 = root / 'sequences' / seq / 'img1'
    fps = float(s['native_fps'])
    rate, rate_src = exact_fps(s['source_video'], fps)
    W, H = s['frame_width'], s['frame_height']
    clip = out_dir / f'{seq}{tag}.mp4'

    encode(img1, clip, n, rate)
    (ok, checks, mapping, worst, shifts, unresolvable, misaligned,
     low_margin) = verify(clip, img1, n, float(Fraction(rate)), W, H)

    lo = s['source_frame_range'][0]
    for m in mapping:
        m['source_frame'] = lo + m['package_frame_1based'] - 1

    record = {
        'clip': clip.name,
        'clip_sha256': (hashlib.sha256(clip.read_bytes()).hexdigest()
                        if clip.exists() else None),
        'package_frames_sha256': s.get('decoded_frames_sha256'),
        'sequence': seq,
        'purpose': 'CVAT VIDEO task vehicle -- forces interpolation/track mode',
        'authoritative': False,
        'canonical_provenance': 'the frozen package frames and '
                                'source_frame_range in manifest.json; this clip '
                                'is a UI vehicle and no measurement derives '
                                'from its pixels',
        'frame_count': n,
        'width': W, 'height': H,
        'native_fps': fps,
        'fps_encoded_as': rate,
        'fps_source': rate_src,
        'frame_mapping_rule': 'CVAT frame k (0-based) = package frame k+1 = '
                              f'source frame {lo} + k',
        'source_frame_range_covered': [mapping[0]['source_frame'],
                                       mapping[-1]['source_frame']] if mapping else None,
        'verification': checks,
        'verified': ok,
        'alignment_test': 'per-frame argmin over offsets -2..+2, plus a '
                          'whole-clip and per-third shift test',
        'shift_test': shifts,
        'frames_with_indistinguishable_neighbours': unresolvable,
        'argmin_failures': misaligned,
        'frames_below_neighbour_margin': low_margin,
        'min_neighbour_margin': round(worst, 3) if worst != float('inf') else None,
        'frame_mapping': mapping,
    }
    (out_dir / f'{seq}{tag}.provenance.json').write_text(
        json.dumps(record, indent=1), encoding='utf-8')

    if not ok:
        clip.unlink(missing_ok=True)
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='data/tracking_val_gt')
    ap.add_argument('--out', default='data/tracking_val_gt/cvat_video')
    ap.add_argument('--smoke', action='store_true',
                    help='build only the 10-frame smoke clip')
    ap.add_argument('--sequence', default='women_1_239',
                    help='sequence for the smoke clip')
    ap.add_argument('--smoke-frames', type=int, default=10)
    args = ap.parse_args()

    if not shutil.which('ffmpeg'):
        raise SystemExit('ffmpeg not found on PATH.')

    root, out = Path(args.root), Path(args.out)
    man = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    seqs = man['sequences']
    if args.smoke:
        seqs = [s for s in seqs if s['sequence'] == args.sequence]
        if not seqs:
            raise SystemExit(f'unknown sequence {args.sequence!r}')

    records = []
    for s in seqs:
        n = args.smoke_frames if args.smoke else s['frame_count']
        tag = f'_smoke{n}' if args.smoke else ''
        print(f'encoding {s["sequence"]}{tag}  {n} frames @ '
              f'{s["native_fps"]} fps ...')

        records.append(build_one(root, s, out, n, tag))

    print(f'\n{"clip":<44}{"frames":>7}{"fps":>12}{"size":>11}{"margin":>8}'
          f'  status')
    for r in records:
        size = f'{r["width"]}x{r["height"]}'
        print(f'  {r["clip"][:42]:<44}{r["frame_count"]:>7}'
              f'{r["fps_encoded_as"]:>12}{size:>11}'
              f'{r["min_neighbour_margin"]:>8}  '
              f'{"VERIFIED" if r["verified"] else "FAILED -- clip deleted"}')
        for c in r['verification']:
            mark = 'ok  ' if c['ok'] else 'FAIL'
            print(f'      [{mark}] {c["check"]}'
                  + (f'   -- {c["detail"]}' if c['detail'] else ''))

    if any(not r['verified'] for r in records):
        raise SystemExit('\nSTOP: a clip did not reproduce its frozen frames.')
    print(f'\nwritten to {out}')
    print('The clip is a UI vehicle. Canonical provenance remains the frozen '
          'package and source-frame mapping.')


if __name__ == '__main__':
    main()
