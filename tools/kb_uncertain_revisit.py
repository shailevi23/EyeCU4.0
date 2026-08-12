#!/usr/bin/env python
"""READ-ONLY: missed_role boxes left on 'uncertain', listed for a second look.

M (non-active match human: bench, coach, ball person, medical, staff) was shown
in the review UI, counted in its own header pill and fully wired server-side --
but the keydown handler had no branch for it, so pressing M did nothing. With no
working key for a coach, U was the only way to park one.

Most of these are probably honest U answers: a real target whose role could not
be read. Some are not. Only the reviewer can tell them apart, so this changes
nothing -- it lists them, and the review UI offers the same list behind REVISIT
U BOXES so each can be re-answered in place.

    python tools/kb_uncertain_revisit.py .
"""
import json
import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
sys.path.insert(0, str(REPO / 'tools'))
import kb_decisions

PKG = REPO / 'experiments/external_sources/keremberke_review'
DEC = PKG / 'decisions.json'
rows = kb_decisions.read_log(DEC)
res = kb_decisions.resolve(DEC)

# every U answered in this pass; the resolver decides which are still open

u = [d for d in rows if d['mode'] in ('missed_role', 'missed_role_manual')
     and d['HUMAN_FINAL_CLASS'] == 'uncertain']
u.sort(key=lambda d: d.get('recorded_utc') or '')
led = {r['BOX_ID']: r for r in
       json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}

print('missed_role / missed_role_manual boxes answered U: %d' % len(u))
print()
print('%-14s %-20s %-9s %s' % ('BOX_ID', 'recorded_utc', 'mode', 'IMAGE'))
out = []
for d in u:
    b = d['BOX_ID']
    r = res[b]
    still = r['disposition'] == 'UNRESOLVED'
    l = led.get(b, {})
    print('%-14s %-20s %-9s %s%s'
          % (b, d.get('recorded_utc'), d['mode'].replace('missed_role', 'mr'),
             l.get('IMAGE', d.get('IMAGE')),
             '' if still else '   (superseded -> %s)' % (r['final_class']
                                                         or r['disposition'])))
    if still and not any(o['BOX_ID'] == b for o in out):
        out.append({'BOX_ID': b, 'IMAGE': l.get('IMAGE', d.get('IMAGE')),
                    'run': l.get('run'), 'bbox_xywh': l.get('bbox_xywh'),
                    'answered_U_utc': d.get('recorded_utc'),
                    'mode': d['mode'],
                    'current_state': 'UNRESOLVED'})

print()
print('still UNRESOLVED and re-answerable with M: %d' % len(out))
p = PKG / 'u_after_failed_M.json'
p.write_text(json.dumps({
    'purpose': ('missed_role boxes answered U. M was advertised in the UI but '
                'the key was never wired, so U was the only way to park a '
                'non-active human. Listed here so they can be re-answered with '
                'M; nothing is changed automatically.'),
    'not_a_correction': True,
    'how_to_resolve': ('open the image in the review UI, click the box, press M '
                       '(or the NON-ACTIVE MATCH HUMAN button) if it really is '
                       'bench/coach/staff; leave it U if the role is genuinely '
                       'unreadable'),
    'count': len(out), 'rows': out}, indent=1), encoding='utf-8')
print('wrote', p.name, '-- nothing was altered')
