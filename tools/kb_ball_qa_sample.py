#!/usr/bin/env python
"""
Draw the BALL QA ROUND 0 sample: 300 images, at random, with no model involved.

Round 0 answers one question -- what fraction of images hold a visible football
that nobody annotated -- and it has to answer it about the DATASET, not about a
detector's opinion of the dataset. So nothing here loads a model, scores an
image, or ranks anything. The only input is the promoted export and a seed.

WHY SIMPLE RANDOM SAMPLING AND NOT STRATIFIED. The first design allocated
proportionally across run x GT-state. It was self-weighting only to within
integer rounding: inclusion probabilities ran 0.2000 to 0.2667 against a mean of
0.2435, because a stratum holding 15 images cannot be given 3.65 of them. That
is a small error, and it is the wrong KIND of error to accept here -- the
estimator would be "positives/300, approximately", and every later reader would
have to decide for themselves whether the approximation mattered.

    SRSWOR, n=300, N=1232, every image at exactly 300/1232.

Then the estimator is positives/300 exactly, and there is no weighting question
to get wrong later. Run, split and GT state are still RECORDED for every sampled
image -- they are just recorded as description, not used as design. The realised
sample will sit near the population proportions because that is what random
sampling does; where it does not, that is sampling variation and correcting it
would be the actual mistake.

The sample is bound to a POPULATION FINGERPRINT: the sha256 of the three
promoted annotation files. If the export changes, the sample is invalid -- not
"probably still fine". An image's ball GT count is part of what the reviewer is
shown, so a sample drawn against different annotations is measuring something
else.

    python tools/kb_ball_qa_sample.py            # write the manifest
    python tools/kb_ball_qa_sample.py --check    # verify without writing

Nothing is applied and the export is never opened for writing.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_images                                                  # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
EXPORT = PKG / 'repaired_export'
MANIFEST = PKG / 'BALL_QA_ROUND0_SAMPLE.json'
SPLITS = ('train', 'valid', 'test')

# Fixed once, recorded, never tuned. A seed chosen after seeing a result is not
# a seed, it is a selection. This one is the ISO date the round was designed.
SEED = 20260814
N_SAMPLE = 300
BALL_NAMES = ('football', 'ball')


def population_fingerprint(export: Path = EXPORT):
    """sha256 per split annotation file, plus one hash over all three.

    This is the identity of the thing being measured. The export manifest
    records the SOURCE hashes and the decisions-log hash, but not a hash of its
    own outputs, so the sample computes one rather than trusting a field that
    does not exist.
    """
    per = {}
    for s in SPLITS:
        p = export / f'{s}_annotations.coco.json'
        if not p.is_file():
            raise FileNotFoundError(f'promoted export is missing {p.name}: {p}')
        per[s] = hashlib.sha256(p.read_bytes()).hexdigest()
    combined = hashlib.sha256(
        ''.join(f'{s}:{per[s]}\n' for s in SPLITS).encode('utf-8')).hexdigest()
    return {'sha256_per_split': per, 'population_sha256': combined}


def _rel(p: Path):
    """Repo-relative where possible; absolute otherwise (e.g. a test tmpdir)."""
    p = Path(p)
    try:
        return str(p.relative_to(REPO)).replace('\\', '/')
    except ValueError:
        return str(p).replace('\\', '/')


def _ball_category(cats):
    for c in cats:
        if c['name'] in BALL_NAMES:
            return c['id']
    raise ValueError('no ball/football category in this split')


def _runs():
    """run label per IMAGE, from the ledger. Descriptive only.

    The ledger is regenerable bulk and may be absent on a fresh checkout. A
    missing run label must not stop the sample being drawn -- it is not part of
    the design, only of the description -- so it degrades to None.
    """
    p = PKG / 'ledger.json'
    if not p.is_file():
        return {}
    out = {}
    for r in json.loads(p.read_text(encoding='utf-8')):
        if r.get('run'):
            out[r['IMAGE']] = r['run']
    return out


def population(export: Path = EXPORT):
    """Every image in the promoted export, with its descriptive metadata.

    Returns a list ordered by (split, file_name) so the enumeration itself is
    deterministic before the seed is ever applied. A population whose ORDER
    depended on dict iteration would make the seed meaningless.
    """
    runs = _runs()
    rows = []
    for s in SPLITS:
        doc = json.loads((export / f'{s}_annotations.coco.json')
                         .read_text(encoding='utf-8'))
        bc = _ball_category(doc['categories'])
        nballs = Counter()
        players = {}
        for a in doc['annotations']:
            if a['category_id'] == bc:
                nballs[a['image_id']] += 1
        # view proxy: median height of non-ball boxes. Small median -> wide or
        # high shot, where a ball is smallest and easiest to miss. Recorded for
        # description; it is NOT a validated classifier and never selects.
        heights = {}
        for a in doc['annotations']:
            if a['category_id'] != bc:
                heights.setdefault(a['image_id'], []).append(float(a['bbox'][3]))
        for im in doc['images']:
            hs = sorted(heights.get(im['id'], []))
            med = hs[len(hs) // 2] if hs else None
            image = f'{s}/{im["file_name"]}'
            rows.append({
                'IMAGE': image,
                'split': s,
                'coco_image_id': im['id'],
                'run': runs.get(image),
                'ball_gt_count': int(nballs.get(im['id'], 0)),
                'gt_state': 'GT0' if not nballs.get(im['id']) else 'GT1+',
                'img_w': im.get('width'),
                'img_h': im.get('height'),
                'median_person_box_h': round(med, 1) if med is not None else None,
                'view_proxy': (None if med is None else
                               ('wide' if med < 60 else 'tight')),
            })
    rows.sort(key=lambda r: (SPLITS.index(r['split']), r['IMAGE']))
    return rows


def draw(pop, n=N_SAMPLE, seed=SEED):
    """SRSWOR via random.sample on a deterministically ordered population.

    random.sample draws WITHOUT replacement and, on a fixed-order sequence with
    a seeded Mersenne Twister, is reproducible across runs and platforms. Every
    element has the same inclusion probability n/N by construction; there is no
    per-item probability to compute or verify because none of them differ.

    The RESULT ORDER is the draw order and is kept as the presentation order,
    so the reviewer meets splits and runs interleaved at random. Presenting in
    population order would align fatigue with dataset grouping: the last hour of
    review would be entirely test-split, entirely one run.
    """
    if n > len(pop):
        raise ValueError(f'cannot draw {n} from a population of {len(pop)}')
    rnd = random.Random(seed)
    return rnd.sample(list(pop), n)


def build(export: Path = EXPORT, n=N_SAMPLE, seed=SEED):
    fp = population_fingerprint(export)
    pop = population(export)
    sample = draw(pop, n, seed)
    return {
        'round': 0,
        'purpose': ('estimate the image-level prevalence of visible but '
                    'unannotated footballs, independently of any model'),
        'primary_endpoint': ('image-level missing-ball defect rate: an image is '
                             'positive if it contains >=1 visible football that '
                             'lacks an annotation'),
        'design': 'simple random sample without replacement (SRSWOR)',
        'model_used': None,
        'no_detector_consulted': True,
        'population': {
            'source': _rel(export),
            'N': len(pop),
            **fp,
        },
        'n': len(sample),
        'seed': seed,
        'inclusion_probability': n / len(pop),
        'inclusion_probability_is_equal_for_every_image': True,
        'presentation_order': ('the seeded draw order, so reviewer fatigue does '
                               'not align with split or run'),
        'descriptive_variables_note': (
            'split, run, ball_gt_count, gt_state and view_proxy are recorded for '
            'post-hoc description only. They were NOT used for inclusion, and '
            'subgroup counts must not be quoted as per-subgroup rates.'),
        'sample': sample,
    }


def _summarise(man):
    s = man['sample']
    print(f'population N = {man["population"]["N"]}   sample n = {man["n"]}   '
          f'seed = {man["seed"]}')
    print(f'inclusion probability = {man["inclusion_probability"]:.9f} '
          f'({man["n"]}/{man["population"]["N"]}) for every image')
    print(f'population fingerprint {man["population"]["population_sha256"][:16]}...')
    pop = population(EXPORT)
    for field in ('split', 'run', 'gt_state', 'view_proxy'):
        cs = Counter(r[field] for r in s)
        cp = Counter(r[field] for r in pop)
        print(f'\nby {field}')
        for k in sorted(cs, key=lambda x: (x is None, x)):
            share = 100 * cs[k] / len(s)
            pshare = 100 * cp[k] / len(pop)
            print(f'  {str(k):<12} {cs[k]:>4}  {share:>5.1f}%   '
                  f'(population {cp[k]:>4}  {pshare:>5.1f}%)')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='rebuild and compare against the stored manifest; '
                         'write nothing')
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--n', type=int, default=N_SAMPLE)
    args = ap.parse_args()

    man = build(EXPORT, args.n, args.seed)
    ok, problems = kb_images.preflight([r['IMAGE'] for r in man['sample']])
    if not ok:
        print(f'REFUSING: {len(problems)} sampled image(s) cannot be resolved '
              f'on disk. A sample containing an unreadable image cannot be '
              f'reviewed at equal inclusion probability.')
        for im, why in problems[:10]:
            print(f'  {im}\n    {why}')
        sys.exit(1)

    if args.check:
        if not MANIFEST.is_file():
            print(f'no stored sample at {MANIFEST}')
            sys.exit(1)
        old = json.loads(MANIFEST.read_text(encoding='utf-8'))
        same_pop = (old['population']['population_sha256']
                    == man['population']['population_sha256'])
        same_ids = ([r['IMAGE'] for r in old['sample']]
                    == [r['IMAGE'] for r in man['sample']])
        print(f'population fingerprint matches: {same_pop}')
        print(f'same 300 images in the same order: {same_ids}')
        if not same_pop:
            print('\nThe promoted export has CHANGED since this sample was '
                  'drawn. The sample is INVALID: draw a new one, and do not '
                  'combine answers across the two.')
        sys.exit(0 if (same_pop and same_ids) else 1)

    if MANIFEST.is_file():
        old = json.loads(MANIFEST.read_text(encoding='utf-8'))
        if ([r['IMAGE'] for r in old['sample']]
                != [r['IMAGE'] for r in man['sample']]):
            print(f'REFUSING TO OVERWRITE {MANIFEST.name}: a different sample '
                  f'is already on record and may already have been reviewed.\n'
                  f'Delete it deliberately if you truly intend to redraw.')
            sys.exit(1)
        print('sample already on record and identical; rewriting is a no-op')

    MANIFEST.write_text(json.dumps(man, indent=1) + '\n', encoding='utf-8')
    _summarise(man)
    print(f'\nwritten: {MANIFEST.relative_to(REPO)}')
    print('no model was loaded, no detector was run, the export was not modified')


if __name__ == '__main__':
    main()
