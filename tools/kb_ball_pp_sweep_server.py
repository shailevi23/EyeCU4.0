#!/usr/bin/env python
"""
PP BALL SWEEP -- the 357 pp images Round 0 never looked at.

Round 0 sampled 300 of 1,232 images at random and found that pp frames carry
almost all of the missing footballs: 55.0% of sampled pp images were missing at
least one, against 7.6% of plain. That is not a rate worth estimating any
further -- at better than one image in two, sampling more would only re-measure
something already established. The remaining pp images are swept in full
instead, because at that density a census is cheaper than an argument.

THE ONTOLOGY IS SETTLED AND THIS TOOL DOES NOT REOPEN IT.

    BALL_DETECTOR_ONTOLOGY      = ALL_VISIBLE_PHYSICAL_FOOTBALLS
    ACTIVE_MATCH_BALL_SELECTION = DOWNSTREAM_TEMPORAL_SELECTOR

So the question here is one question, not two:

    Is there any REAL PHYSICAL FOOTBALL visible that is not annotated?

Active or spare, ball-boy or behind the goal -- all of them are football. The
reviewer is never asked which kind, because under this policy the answer would
change nothing about the annotation. That distinction is recorded elsewhere as
provenance and the temporal selector deals with it downstream.

WHAT IS NOT IN THE QUEUE. The 129 pp images Round 0 already reviewed. Their 128
findings are frozen and will be combined with this sweep when a corrected export
is built; re-reviewing them would duplicate human work and, worse, would let a
second answer silently supersede a frozen one. Plain images are not here either
-- that strategy is decided after this sweep reports.

    python tools/kb_ball_pp_sweep_server.py

Nothing is applied, no annotation is modified, no detector is loaded.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_ball_ontology_revisit_server as ONTO                    # noqa: E402
import kb_ball_qa_sample as kb_sample                             # noqa: E402
import kb_ball_qa_server as QA                                    # noqa: E402
import kb_decisions                                               # noqa: E402
import kb_images                                                  # noqa: E402
from kb_missing_target_server import validate_bbox                # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
DECISIONS = PKG / 'decisions.json'
POLICY = PKG / 'BALL_ONTOLOGY_POLICY.json'
QUEUE = PKG / 'BALL_PP_SWEEP_QUEUE.json'
LOCK = threading.Lock()

SWEEP_MODE = 'ball_pp_sweep'
ANSWERS = ('NO_MISSING_BALL', 'MISSING_BALL', 'UNSURE')

# pp membership is read from the FILENAME, not the ledger run label. One image
# (536_pp_jpg...) is a pp frame whose ledger row carries no run, so a run-based
# population would be 485 and would silently drop it. The policy says "all pp
# images", and the filename is what makes an image a pp image.
PP_MARK = '_pp_'


def is_pp(image: str) -> bool:
    return PP_MARK in image.rsplit('/', 1)[-1]


def round0_images():
    man = json.loads(kb_sample.MANIFEST.read_text(encoding='utf-8'))
    return {r['IMAGE'] for r in man['sample']}


def queue_images(export: Path = None):
    """Every pp image, minus every image Round 0 already reviewed.

    Deterministic: population order is the sampler's own (split, filename)
    ordering, so the same inputs always give the same queue in the same order.
    """
    pop = kb_sample.population(export or kb_sample.EXPORT)
    seen = round0_images()
    return [r for r in pop if is_pp(r['IMAGE']) and r['IMAGE'] not in seen]


def build_queue(export: Path = None):
    pop = kb_sample.population(export or kb_sample.EXPORT)
    seen = round0_images()
    pp = [r for r in pop if is_pp(r['IMAGE'])]
    q = queue_images(export)
    return {
        'queue': 'ball_pp_sweep',
        'purpose': ('full human sweep of every pp image Round 0 did not '
                    'review, under the ALL_VISIBLE_PHYSICAL_FOOTBALLS policy'),
        'ontology': json.loads(POLICY.read_text(encoding='utf-8'))[
            'BALL_DETECTOR_ONTOLOGY'],
        'design': ('census, not a sample: no inclusion probability and no '
                   'confidence interval applies to a complete enumeration'),
        'population': {
            'N_all_images': len(pop),
            'N_pp_images': len(pp),
            'pp_definition': f'"{PP_MARK}" in the file name',
            **kb_sample.population_fingerprint(export or kb_sample.EXPORT),
        },
        'round0_pp_already_reviewed': len(pp) - len(q),
        'n': len(q),
        'no_detector_consulted': True,
        'images': [{'IMAGE': r['IMAGE'], 'split': r['split'], 'run': r['run'],
                    'ball_gt_count': r['ball_gt_count'],
                    'gt_state': r['gt_state'], 'img_w': r['img_w'],
                    'img_h': r['img_h'], 'view_proxy': r['view_proxy'],
                    'coco_image_id': r['coco_image_id']} for r in q],
    }


def answers(decisions: Path = DECISIONS):
    """Effective sweep answer per IMAGE. Latest event wins; nothing rewritten."""
    per = {}
    for d in kb_decisions.read_log(decisions):
        if d.get('mode') == SWEEP_MODE:
            per.setdefault(d['IMAGE'], []).append(d)
    out = {}
    for im, evs in per.items():
        evs = sorted(evs, key=lambda d: (d.get('recorded_utc') or '', d['_line']))
        win = evs[-1]
        out[im] = {
            'answer': win.get('answer'),
            'missing': win.get('missing_balls') or [],
            'recorded_utc': win.get('recorded_utc'),
            'history': [{'answer': e.get('answer'),
                         'n_missing': len(e.get('missing_balls') or []),
                         'recorded_utc': e.get('recorded_utc')} for e in evs],
        }
    return out


PAGE = ONTO.PAGE          # replaced in main(); kept so imports do not fail


def build_state(show_context=False, decisions: Path = DECISIONS):
    if not QUEUE.is_file():
        raise SystemExit(f'no queue at {QUEUE}; run with --build-queue first')
    q = json.loads(QUEUE.read_text(encoding='utf-8'))
    live = kb_sample.population_fingerprint()['population_sha256']
    if live != q['population']['population_sha256']:
        raise SystemExit(
            'REFUSING: the promoted export has changed since this queue was '
            'built. Rebuild it rather than reviewing a stale population.')
    ans = answers(decisions)
    flags = ONTO.gt_flags(decisions)
    cache = {}
    items = []
    for r in q['images']:
        split = r['split']
        if split not in cache:
            doc = json.loads((kb_sample.EXPORT /
                              f'{split}_annotations.coco.json')
                             .read_text(encoding='utf-8'))
            bc = kb_sample._ball_category(doc['categories'])
            per = {}
            for a in doc['annotations']:
                per.setdefault(a['image_id'], []).append(a)
            cache[split] = (bc, per)
        bc, per = cache[split]
        rows = per.get(r['coco_image_id'], [])
        a = ans.get(r['IMAGE'], {})
        items.append({
            'IMAGE': r['IMAGE'], 'split': split, 'run': r['run'],
            'img_w': r['img_w'], 'img_h': r['img_h'],
            'gt_state': r['gt_state'],
            'ball_gt': [{'bbox': x['bbox'], 'annotation_id': x['id'],
                         'BOX_ID': f'{split}:{x["id"]}', 'cls': 'football',
                         'flag': flags.get(f'{split}:{x["id"]}', {}).get(
                             'flag_type')}
                        for x in rows if x['category_id'] == bc],
            'context': [{'bbox': x['bbox']} for x in rows
                        if x['category_id'] != bc],
            'answer': a.get('answer'),
            'missing': a.get('missing', []),
            'history': a.get('history', []),
        })
    return {'items': items, 'show_context': show_context,
            'FALSE_BALL': ONTO.FALSE_BALL, 'BAD_BOX': ONTO.BAD_BOX,
            'EXISTING_NON_ACTIVE': ONTO.EXISTING_NON_ACTIVE,
            'flag_counts': ONTO.flag_counts(decisions),
            'ontology': q['ontology'],
            'source_log': kb_decisions.log_version(decisions)}


def append(rec):
    with LOCK:
        with open(DECISIONS, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + '\n')
            fh.flush()


def build_id_info():
    page_sha = hashlib.sha256(PAGE.encode('utf-8')).hexdigest()
    return {'build': page_sha[:12], 'page_sha256': page_sha,
            'tool': 'kb_ball_pp_sweep_server'}


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    SHOW_CONTEXT = False
    IMAGES = frozenset()
    DIMS = {}
    BALL_GT = {}

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        v = memoryview(body)
        for i in range(0, len(v), 1 << 16):
            self.wfile.write(v[i:i + (1 << 16)])
        self.wfile.flush()

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        if p == '/':
            return self._send(200,
                              PAGE.replace('__BUILD__',
                                           build_id_info()['build'])
                              .encode('utf-8'), 'text/html; charset=utf-8')
        if p == '/api/build':
            return self._send(200, json.dumps(build_id_info()).encode())
        if p == '/api/state':
            return self._send(200, json.dumps(
                build_state(self.SHOW_CONTEXT)).encode())
        if p.startswith('/img/'):
            want = p[len('/img/'):]
            if want not in self.IMAGES:
                return self._send(403, json.dumps(
                    {'error': 'not in the pp sweep queue', 'IMAGE': want}
                ).encode())
            try:
                body, ctype = kb_images.read(want)
            except kb_images.ImageError as e:
                print(f'IMAGE 404  {want}  --  {e}', flush=True)
                return self._send(404, json.dumps(
                    {'error': str(e), 'IMAGE': want}).encode())
            return self._send(200, body, ctype)
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        route = urlparse(self.path).path
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')

        if route == '/api/flag':
            # The blue-GT side channel is shared with the ontology tool. Its
            # handler is called with THIS handler as self -- both classes expose
            # BALL_GT and _send with the same shape, and reimplementing the
            # validation here is how the first server and its auditor came to
            # disagree about what a click meant. ONTO.DECISIONS is pointed at
            # the same log for the call, since _flag reads it for retraction.
            saved, ONTO.DECISIONS = ONTO.DECISIONS, DECISIONS
            try:
                return ONTO.H._flag(self, d)
            finally:
                ONTO.DECISIONS = saved

        if route != '/api/answer':
            return self._send(404, b'', 'text/plain')

        image = str(d.get('IMAGE', ''))
        if image not in self.IMAGES:
            return self._send(400, json.dumps(
                {'error': 'image is not in the pp sweep queue; the 129 pp '
                          'images Round 0 already reviewed are deliberately '
                          'excluded and must not be answered again here',
                 'IMAGE': image}).encode())
        ans = d.get('answer')
        if ans not in ANSWERS:
            return self._send(400, json.dumps(
                {'error': f'answer must be one of {ANSWERS}'}).encode())

        w, h = self.DIMS.get(image, (None, None))
        missing = []
        if ans == 'MISSING_BALL':
            raw = d.get('missing_balls_xywh') or []
            if not raw:
                return self._send(400, json.dumps(
                    {'error': 'MISSING_BALL requires at least one drawn ball; '
                              'use UNSURE if it cannot be located'}).encode())
            for b in raw:
                bbox, err = validate_bbox(b, w, h)
                if err:
                    return self._send(400, json.dumps({'error': err}).encode())
                missing.append({'bbox_xywh': bbox,
                                'geometry_author': 'human drawn',
                                'coordinate_space': 'original image pixels'})

        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        append({
            'mode': SWEEP_MODE,
            'BOX_ID': f'PPSWEEP:{image}',
            'IMAGE': image,
            'answer': ans,
            'HUMAN_FINAL_CLASS': None,
            'image_is_positive': ans == 'MISSING_BALL',
            'missing_balls': missing,
            'n_missing_objects': len(missing),
            'img_w': w, 'img_h': h,
            'ontology': 'ALL_VISIBLE_PHYSICAL_FOOTBALLS',
            'no_model_proposal_used': True,
            'no_detector_consulted': True,
            'recorded_utc': now, 'author': 'human reviewer',
        })
        return self._send(200, json.dumps(
            {'ok': True, 'missing': missing, 'recorded_utc': now}).encode())


def _page():
    """The Round-0 review page, relabelled for this queue.

    The Round-0 UI is the one that has been driven, fixed and tested -- the
    tile sweep, the multi-ball drawing, the zoom and the coordinate handling
    all work there. Forking it would mean re-earning that, so it is reused and
    only its wording changes. The blue-GT observation panel comes across too.
    """
    p = QA.PAGE
    p = p.replace('<title>ball QA round 0</title>',
                  '<title>pp ball sweep</title>')
    p = p.replace('<b>BALL QA ROUND 0</b>',
                  '<b>PP BALL SWEEP</b> <span class="pill" style="font:10px '
                  'monospace;color:#777">build __BUILD__</span>')
    p = p.replace('image ${i+1}/${S.items.length}',
                  'image ${i+1}/${S.items.length}')
    p = p.replace(
        'Is there any visible football in this image that is NOT\n  annotated?',
        'Is there any REAL PHYSICAL FOOTBALL visible that is NOT annotated?')
    p = p.replace('Is there any visible football in this image that is NOT',
                  'Is there any REAL PHYSICAL FOOTBALL visible that is NOT')
    return p


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--build-queue', action='store_true',
                    help='(re)write the queue manifest, then exit')
    ap.add_argument('--port', type=int, default=8746)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--show-context', action='store_true')
    args = ap.parse_args()

    if args.build_queue or not QUEUE.is_file():
        q = build_queue()
        QUEUE.write_text(json.dumps(q, indent=1) + '\n', encoding='utf-8')
        print(f'queue written: {QUEUE.relative_to(REPO)}')
        print(f'  pp images {q["population"]["N_pp_images"]}  '
              f'already reviewed in Round 0 {q["round0_pp_already_reviewed"]}  '
              f'queue {q["n"]}')
        if args.build_queue:
            return

    global PAGE
    PAGE = _page()

    q = json.loads(QUEUE.read_text(encoding='utf-8'))
    H.SHOW_CONTEXT = args.show_context
    H.IMAGES = frozenset(r['IMAGE'] for r in q['images'])
    H.DIMS = {r['IMAGE']: (r['img_w'], r['img_h']) for r in q['images']}

    # blue-GT observations are offered on the sweep images too
    gt = {}
    cache = {}
    for r in q['images']:
        split = r['split']
        if split not in cache:
            doc = json.loads((kb_sample.EXPORT /
                              f'{split}_annotations.coco.json')
                             .read_text(encoding='utf-8'))
            bc = kb_sample._ball_category(doc['categories'])
            per = {}
            for a in doc['annotations']:
                if a['category_id'] == bc:
                    per.setdefault(a['image_id'], []).append(a)
            cache[split] = per
        for a in cache[split].get(r['coco_image_id'], []):
            gt[f'{split}:{a["id"]}'] = {
                'IMAGE': r['IMAGE'], 'split': split, 'annotation_id': a['id'],
                'bbox_xywh': a['bbox']}
    H.BALL_GT = gt

    ok, problems = kb_images.preflight(sorted(H.IMAGES))
    if not ok:
        print(f'REFUSING TO START: {len(problems)} image(s) cannot be resolved.')
        for im, why in problems[:10]:
            print(f'  {im}\n    {why}')
        sys.exit(1)

    ans = answers()
    done = sum(1 for im in H.IMAGES if ans.get(im, {}).get('answer'))
    pos = sum(1 for im in H.IMAGES
              if ans.get(im, {}).get('answer') == 'MISSING_BALL')
    found = sum(len(ans[im]['missing']) for im in H.IMAGES
                if im in ans and ans[im]['answer'] == 'MISSING_BALL')
    print(f'PP BALL SWEEP -- {len(H.IMAGES)} pp images not reviewed in Round 0')
    print(f'ontology: {q["ontology"]} '
          f'(active vs spare is NOT asked here)')
    print(f'{done} answered, {len(H.IMAGES) - done} outstanding, '
          f'{pos} positive so far, {found} footballs drawn')
    print(f'{len(H.BALL_GT)} existing ball annotations in these images')
    print('preflight: every image resolves')
    print('\nRound 0 stays frozen: its 300 images and 128 findings are not '
          'in this queue.')
    print('NO DETECTOR IS LOADED. No annotation is modified.')
    print('Keys: 1 none  2 missing (draw)  3 unsure  N next  B prev  T tiles')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
