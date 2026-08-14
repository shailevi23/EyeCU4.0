#!/usr/bin/env python
"""
Freeze and report BALL QA ROUND 0.

The primary endpoint is the IMAGE-LEVEL missing-ball defect rate. An image is
positive if it holds at least one visible football that lacks an annotation. An
image with three missing balls is ONE positive image and THREE objects, and this
tool never adds those two numbers or divides one by the other.

WHY THERE IS NO "OBJECT MISSING RATE". A rate needs a denominator drawn from a
known population. Images have one: 300 were drawn from 1,232 at a known
inclusion probability. Unannotated balls do not -- there is no enumerable
population of them to sample from, and the number of balls that COULD have been
missed in an image is unknowable. So objects are reported as counts and as a
size distribution, and any ratio built from them would be a number with no
sampling interpretation dressed up as a measurement.

THE INTERVAL MATCHES THE DESIGN. 300 images were drawn WITHOUT replacement from
a finite population of 1,232, so the sampling distribution of the positive count
is hypergeometric, not binomial. The primary interval is therefore obtained by
inverting that distribution: it is the set of population positive-counts M for
which the observed x is not in either 2.5% tail. Its endpoints are integer
counts of images out of 1,232 -- which is what the estimand actually is. There
are only 1,233 candidate values of M, so the inversion is an exact search, not
an approximation.

Clopper-Pearson is still reported, but labelled for what it is: a CONSERVATIVE
BINOMIAL REFERENCE that pretends the population is infinite. It is uniformly
wider (at x=0, [0, 1.22%] against the finite-population [0, 1.06%]) because it
discards the information that 300 of the 1,232 images were actually inspected.
Calling it "the exact interval" for this design would be wrong.

NO PREVALENCE INTERVAL BEFORE THE ROUND IS COMPLETE. --interim prints counts
only. An interval computed from 40 answers is not an early view of the result;
it is a different quantity with a denominator that does not exist yet, and a
number printed beside the word "CI" gets quoted no matter how it is captioned.

    python tools/kb_ball_round0_report.py              # freeze, once complete
    python tools/kb_ball_round0_report.py --interim    # counts only, no CI

The report embeds three fingerprints -- the export, the sample manifest and the
decision log -- so a later reader can tell whether it still describes the
current state. A report that cannot prove what it was built from is exactly the
failure the missing-target queue had when it read "0 flags" with 51 on record.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_ball_qa_sample                                          # noqa: E402
import kb_ball_qa_server                                          # noqa: E402
import kb_decisions                                               # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
DECISIONS = PKG / 'decisions.json'
REPORT = PKG / 'BALL_QA_ROUND0_RESULT.json'
BUCKETS = ((5, '<=5px'), (8, '<=8px'), (12, '<=12px'), (float('inf'), '>12px'))


def clopper_pearson(k, n, alpha=0.05):
    """Binomial CI. Conservative REFERENCE only -- not this design's interval.

    Correct when sampling with replacement from an infinite population. Here it
    ignores that 300 of 1,232 images were inspected, so it is uniformly wider
    than the truth. Reported for comparability, never as the primary result.
    """
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def hypergeometric_ci(x, N, n, alpha=0.05):
    """Exact SRSWOR interval for M, the number of positive images in the
    population of N. Returns (M_lo, M_hi) as integer image counts.

    Inversion of the hypergeometric sampling distribution:

        M_lo = min{ M : P(X >= x | N, M, n) >= alpha/2 }
        M_hi = max{ M : P(X <= x | N, M, n) >= alpha/2 }

    i.e. every M kept in the interval is one under which the observed x would
    not be a 2.5%-tail event. Both probabilities are monotone in M, so a linear
    scan over the 0..N candidates finds the exact endpoints -- no root-finding
    and no continuity correction.

    The estimand is a COUNT of images out of N, and this interval reports it as
    one. A consequence worth understanding rather than patching: M_lo can exceed
    x. At x=3, N=1232, n=300 the bound is M=4, because if the population held
    exactly 3 positives, drawing all 3 of them in a 300-image sample has
    probability 0.0143 -- itself a 2.5%-tail event. Observing 3 is mild evidence
    that more than 3 exist.
    """
    from scipy.stats import hypergeom
    if not (0 <= x <= n <= N):
        raise ValueError(f'need 0 <= x <= n <= N, got x={x} n={n} N={N}')
    lo = N
    for M in range(0, N + 1):
        if float(hypergeom.sf(x - 1, N, M, n)) >= alpha / 2:
            lo = M
            break
    hi = x
    for M in range(N, -1, -1):
        if float(hypergeom.cdf(x, N, M, n)) >= alpha / 2:
            hi = M
            break
    return int(lo), int(hi)


def size_bucket(w):
    for lim, name in BUCKETS:
        if w <= lim:
            return name
    return '>12px'


def collect(sample_path=None, decisions=DECISIONS):
    """Everything the report needs, plus the reasons it might refuse."""
    sp = Path(sample_path) if sample_path else kb_ball_qa_sample.MANIFEST
    if not sp.is_file():
        return None, [f'no Round-0 sample manifest at {sp}']
    man = json.loads(sp.read_text(encoding='utf-8'))
    blocking = []

    live = kb_ball_qa_sample.population_fingerprint()
    pop_ok = live['population_sha256'] == man['population']['population_sha256']
    if not pop_ok:
        blocking.append(
            'the promoted export has changed since the sample was drawn: '
            f'sample {man["population"]["population_sha256"][:16]}... vs live '
            f'{live["population_sha256"][:16]}... -- the answers and the '
            'population no longer describe the same thing')

    ans = kb_ball_qa_server.answers(decisions)
    sampled = [r['IMAGE'] for r in man['sample']]
    inset = set(sampled)

    # An answer recorded for an image outside the sample would break equal
    # inclusion probability: the reviewed set would no longer be the drawn set,
    # and positives/300 would stop estimating the population rate.
    stray = sorted(im for im in ans if im not in inset)
    if stray:
        blocking.append(
            f'{len(stray)} Round-0 answer(s) exist for images outside the '
            f'sample, e.g. {stray[:3]} -- the reviewed set is not the drawn '
            f'set, so the unweighted interval would not be valid')

    answered = {im: ans[im] for im in sampled if im in ans}
    unanswered = [im for im in sampled if im not in ans]
    unsure = [im for im, a in answered.items() if a['answer'] == 'UNSURE']
    positives = [im for im, a in answered.items() if a['answer'] == 'MISSING_BALL']
    negatives = [im for im, a in answered.items()
                 if a['answer'] == 'NO_MISSING_BALL']
    return {'manifest': man, 'manifest_path': sp, 'answers': answered,
            'sampled': sampled, 'unanswered': unanswered, 'unsure': unsure,
            'positives': positives, 'negatives': negatives,
            'population_fingerprint_matches': pop_ok,
            'live_fingerprint': live}, blocking


def build(data):
    man = data['manifest']
    n = man['n']
    meta = {r['IMAGE']: r for r in man['sample']}
    pos, unsure, unans = data['positives'], data['unsure'], data['unanswered']
    k = len(pos)

    objects = []
    for im in pos:
        for b in data['answers'][im]['missing']:
            x, y, w, h = b['bbox_xywh']
            objects.append({'IMAGE': im, 'bbox_xywh': b['bbox_xywh'],
                            'width': w, 'height': h,
                            'size_bucket': size_bucket(w),
                            'split': meta[im]['split'], 'run': meta[im]['run'],
                            'gt_state': meta[im]['gt_state']})

    N = man['population']['N']
    m_lo, m_hi = hypergeometric_ci(k, N, n)
    b_lo, b_hi = clopper_pearson(k, n)
    per_image = Counter(len(data['answers'][im]['missing']) for im in pos)

    def cut(field, images):
        return dict(Counter(meta[im][field] for im in images))

    resolved = n - len(unans) - len(unsure)
    return {
        'round': 0,
        'frozen': not unans and not unsure,
        'design': 'simple random sample without replacement (SRSWOR)',
        'no_detector_consulted': True,
        'fingerprints': {
            'population_sha256': man['population']['population_sha256'],
            'population_matches_live_export': data['population_fingerprint_matches'],
            'sample_manifest_sha256': hashlib.sha256(
                data['manifest_path'].read_bytes()).hexdigest(),
            'decisions_log': kb_decisions.log_version(DECISIONS),
        },
        'primary': {
            'endpoint': 'image-level missing-ball defect rate',
            'definition': ('an image is positive if it contains >=1 visible '
                           'football that lacks an annotation'),
            'positive_images': k,
            'n': n,
            'N': N,
            'rate': k / n,
            'ci_method': ('exact finite-population interval, by inversion of '
                          'the hypergeometric sampling distribution for SRSWOR '
                          f'(N={N}, n={n})'),
            'ci95_finite_population': [m_lo / N, m_hi / N],
            'ci95_population_counts': [m_lo, m_hi],
            'ci95_counts_note': ('endpoints are integer numbers of positive '
                                 f'IMAGES out of {N}, which is what the '
                                 'estimand is; the rates above are those counts '
                                 f'divided by {N}'),
            'conservative_binomial_reference': {
                'ci95_clopper_pearson': [b_lo, b_hi],
                'note': ('reference only. Clopper-Pearson assumes sampling with '
                         'replacement from an infinite population, so it '
                         'discards the finite-population information and is '
                         'uniformly wider. It is NOT the exact interval for '
                         'this design.'),
            },
            'estimator_note': ('every image had inclusion probability exactly '
                               f'{man["inclusion_probability"]:.9f}, so the '
                               'unweighted proportion is the estimator with no '
                               'weighting adjustment'),
        },
        'secondary': {
            'total_missing_objects': len(objects),
            'objects_per_positive_image': dict(sorted(per_image.items())),
            'size_buckets': dict(Counter(o['size_bucket'] for o in objects)),
            'no_object_rate_reported': (
                'deliberate: unannotated balls have no enumerable population, '
                'so an object-level rate would have no denominator and no '
                'sampling interpretation'),
            'objects': objects,
        },
        'descriptive': {
            'note': ('post-hoc cuts of a sample that was NOT allocated by these '
                     'variables. Counts are incidental and must not be quoted '
                     'as per-subgroup rates or used for between-group '
                     'comparison.'),
            'positives_by_split': cut('split', pos),
            'positives_by_run': cut('run', pos),
            'positives_by_gt_state': cut('gt_state', pos),
            'positives_by_view_proxy': cut('view_proxy', pos),
            'sample_by_split': cut('split', data['sampled']),
            'sample_by_run': cut('run', data['sampled']),
            'sample_by_gt_state': cut('gt_state', data['sampled']),
        },
        # Logical bounds, not inference: what the final count COULD be once the
        # outstanding and UNSURE images resolve. No probability is attached and
        # none should be -- these are the arithmetic extremes.
        'interim_bounds': {
            'min_possible_positives': k,
            'max_possible_positives': k + len(unans) + len(unsure),
            'note': ('logical bounds, not a confidence interval: no sampling '
                     'probability is attached to either endpoint'),
        },
        'unresolved': {
            'unsure_images': len(unsure),
            'unanswered_images': len(unans),
            'resolved_images': resolved,
            'interpretation': _unresolved_note(len(unsure), len(unans), n, k),
            'unsure_list': unsure,
        },
        'escalation': _escalate(k, objects, len(unsure)),
    }


def _unresolved_note(n_unsure, n_unans, n, k):
    if not n_unsure and not n_unans:
        return ('all 300 images have a definite answer; the denominator is the '
                'full sample')
    parts = []
    if n_unans:
        parts.append(f'{n_unans} image(s) not yet answered')
    if n_unsure:
        parts.append(f'{n_unsure} image(s) answered UNSURE')
    lo_k, hi_k = k, k + n_unsure + n_unans
    return (f'{" and ".join(parts)}. The denominator of 300 must NOT be used as '
            f'though every outcome were known. UNSURE is not evidence of '
            f'absence: the true positive count lies between {lo_k} and {hi_k}, '
            f'so the reported rate is a LOWER BOUND and the upper confidence '
            f'limit is understated.')


def _escalate(k, objects, n_unsure):
    """The rule, fixed before any review. Reported, never executed."""
    small = [o for o in objects if o['width'] <= 8]
    if n_unsure > 15:
        band = 'UNSURE>15'
        action = ('review the instrument and the guidance before quoting a '
                  'prevalence; a high UNSURE rate makes the positive rate a '
                  'lower bound and the interval unreliable')
    elif k == 0:
        band = '0 positives'
        action = ('stop Round 0; report the CI; no Round 1 required; '
                  'Experiment D may proceed')
    elif k <= 3 and not small:
        band = '1-3 positives, all found balls >12px'
        action = ('correct later if desired; no Round 1. The defensible '
                  'statement is "no evidence from Round 0 of a tiny-ball-'
                  'specific missing-label problem". DO NOT claim a cause -- '
                  'why these were missed is unknown without a separate '
                  'investigation')
    elif k <= 3:
        band = '1-3 positives with at least one found ball <=8px'
        action = 'extend the measurement to 600 before deciding'
    elif k <= 9:
        band = '4-9 positives'
        action = 'extend to 600 and trigger investigation / Round 1'
    else:
        band = '>=10 positives'
        action = ('systematic issue: stop and characterise it before '
                  'Experiment D')
    return {'band': band, 'pre_declared_action': action,
            'note': 'Round 1 is NOT started automatically; return this result '
                    'to the human first'}


def _header(rep):
    f = rep['fingerprints']
    print(f'design       : {rep["design"]}, no detector consulted')
    print(f'population   : {f["population_sha256"][:16]}...  '
          f'matches live export: {f["population_matches_live_export"]}')
    print(f'decisions log: {f["decisions_log"]["decisions_sha256"][:16]}...  '
          f'{f["decisions_log"]["decisions_lines"]} lines')


def _print_interim(rep):
    """Counts only. No prevalence estimate and no interval of any kind.

    A partial sample has no denominator: the 260 images not yet looked at are
    not negatives, and an interval built as though they were would be measuring
    the images that happened to be reviewed first. Worse, a number printed next
    to "CI" gets quoted regardless of the caption around it -- so the honest
    move is not to compute one until the round is complete.
    """
    p, s, u = rep['primary'], rep['secondary'], rep['unresolved']
    n = p['n']
    answered = n - u['unanswered_images']
    print('=' * 72)
    print('BALL QA ROUND 0 -- INTERIM STATUS')
    print('=' * 72)
    _header(rep)
    print('\nPROGRESS (counts only -- no prevalence estimate is available yet)')
    print(f'  answered           : {answered} / {n}')
    print(f'  outstanding        : {u["unanswered_images"]}')
    print(f'  positive images    : {p["positive_images"]}')
    print(f'  missing objects    : {s["total_missing_objects"]}')
    print(f'  UNSURE             : {u["unsure_images"]}')
    b = rep['interim_bounds']
    print(f'\nLOGICAL BOUNDS on the final positive count (not a confidence '
          f'interval)')
    print(f'  minimum possible   : {b["min_possible_positives"]}   '
          f'(every remaining image resolves negative)')
    print(f'  maximum possible   : {b["max_possible_positives"]}   '
          f'(every outstanding and UNSURE image resolves positive)')
    print(f'  {b["note"]}')
    print(f'\nNO CONFIDENCE INTERVAL IS REPORTED HERE. A Round-0 prevalence '
          f'estimate\nexists only once all {n} images have an effective '
          f'outcome; the frozen report\nis the first place it may appear.')


def _print_frozen(rep):
    p = rep['primary']
    print('=' * 72)
    print('BALL QA ROUND 0 -- FROZEN RESULT')
    print('=' * 72)
    _header(rep)
    print(f'\nPRIMARY -- {p["endpoint"]}')
    print(f'  positive images    : {p["positive_images"]} / {p["n"]}')
    print(f'  estimated prevalence: {100 * p["rate"]:.2f}%')
    lo, hi = p['ci95_population_counts']
    rlo, rhi = p['ci95_finite_population']
    print(f'  exact 95% CI (finite population, SRSWOR N={p["N"]} n={p["n"]}):')
    print(f'      [{100 * rlo:.2f}%, {100 * rhi:.2f}%]   '
          f'= [{lo}, {hi}] positive images out of {p["N"]}')
    b = p['conservative_binomial_reference']['ci95_clopper_pearson']
    print(f'  conservative binomial reference (Clopper-Pearson, not this '
          f'design):')
    print(f'      [{100 * b[0]:.2f}%, {100 * b[1]:.2f}%]')
    s = rep['secondary']
    print(f'\nSECONDARY (counts, never a rate)')
    print(f'  missing objects    : {s["total_missing_objects"]}')
    print(f'  objects per positive image: {s["objects_per_positive_image"] or "-"}')
    print(f'  size buckets       : {s["size_buckets"] or "-"}')
    u = rep['unresolved']
    print(f'\nUNRESOLVED')
    print(f'  UNSURE {u["unsure_images"]}   unanswered {u["unanswered_images"]}'
          f'   resolved {u["resolved_images"]}')
    print(f'  {u["interpretation"]}')
    d = rep['descriptive']
    print(f'\nDESCRIPTIVE (post-hoc; not per-subgroup rates)')
    print(f'  positives by split : {d["positives_by_split"] or "-"}')
    print(f'  positives by run   : {d["positives_by_run"] or "-"}')
    print(f'  positives by GT    : {d["positives_by_gt_state"] or "-"}')
    e = rep['escalation']
    print(f'\nPRE-DECLARED ESCALATION BAND: {e["band"]}')
    print(f'  {e["pre_declared_action"]}')
    print(f'  {e["note"]}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--interim', action='store_true',
                    help='report status while the review is still in progress; '
                         'writes nothing and is not a result')
    ap.add_argument('--out', default=str(REPORT))
    args = ap.parse_args()

    data, blocking = collect()
    if blocking:
        print('REFUSING TO REPORT:')
        for b in blocking:
            print(f'  - {b}')
        sys.exit(2)

    rep = build(data)
    incomplete = data['unanswered'] or data['unsure']
    if incomplete and not args.interim:
        print('REFUSING TO FREEZE A RESULT:')
        if data['unanswered']:
            print(f'  - {len(data["unanswered"])} of {rep["primary"]["n"]} '
                  f'sampled images have no answer')
        if data['unsure']:
            print(f'  - {len(data["unsure"])} image(s) answered UNSURE and '
                  f'remain unresolved')
        print('\nEvery sampled image needs a definite answer before a rate can '
              'be quoted, because an unanswered image is not a negative one.\n'
              'Run with --interim for status without a result.')
        sys.exit(3)

    if args.interim:
        _print_interim(rep)
        print('\n--interim: nothing written. This is a status, not a result.')
        return
    _print_frozen(rep)
    Path(args.out).write_text(json.dumps(rep, indent=1) + '\n', encoding='utf-8')
    print(f'\nfrozen result written: {Path(args.out).relative_to(REPO)}')
    print('the export was not modified and no correction was applied; '
          'findings stay in the measured rate even after they are corrected')


if __name__ == '__main__':
    main()
