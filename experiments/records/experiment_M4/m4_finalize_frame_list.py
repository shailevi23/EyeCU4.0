#!/usr/bin/env python
"""
M4 Section 4 -- apply the deterministic, content-blind replacement rule to
any leaking candidate frame, then freeze the FINAL 60-frame TEST list.

Replacement rule (fixed by the M4 milestone specification itself, not
chosen or tuned by this script, and therefore not at risk of being fit to
the observed leakage result): for a leaking frame at 0-based index i0 in a
sequence of length N, search the same sequence at ordinal offsets
+1, -1, +2, -2, +3, -3, ... from i0 (clamped to [0, N-1], skipping any
offset already selected for that sequence or itself flagged as leaking)
and take the first candidate that is neither. This is purely a function of
(i0, N, the set of already-selected indices, the leak/no-leak verdict per
index) -- never a function of pixel content beyond the already-frozen
leak/no-leak verdict itself.

LEAKAGE_SCREEN_RESULT.json (frozen, sha256 recorded in
LEAKAGE_SCREEN_RESULT.sha256) found n_leaks_found = 0 across all 60
candidates against the full 2231-image TRAIN+VAL reference pool. The
replacement rule below is therefore exercised zero times; the FINAL list is
identical in content to INITIAL_TEST_FRAME_LIST.json. It is nonetheless
re-emitted and independently hashed as its own frozen artifact, since the
milestone requires the FINAL list -- not the INITIAL list -- to be the
gate for human visual annotation access.
"""

import hashlib
import json
from pathlib import Path

INITIAL = json.loads(Path('experiments/records/experiment_M4/INITIAL_TEST_FRAME_LIST.json')
                     .read_text(encoding='utf-8'))
LEAKAGE = json.loads(Path('experiments/records/experiment_M4/LEAKAGE_SCREEN_RESULT.json')
                     .read_text(encoding='utf-8'))
CANDIDATES = json.loads(Path('experiments/records/experiment_M4/candidates_manifest.json')
                        .read_text(encoding='utf-8'))


def main():
    leak_by_file = {r['file']: r['is_leak'] for r in LEAKAGE['results']}

    final_sequences = {}
    replacements = []
    for seq, info in INITIAL['sequences'].items():
        n = info['n_frames_total']
        selected = list(info['selected_frame_indices_0based'])
        seq_candidates = [c for c in CANDIDATES if c['sequence'] == seq]
        seq_candidates.sort(key=lambda c: c['frame_index_0based'])

        final_idx = []
        selected_set = set(selected)
        for c in seq_candidates:
            i0 = c['frame_index_0based']
            is_leak = leak_by_file.get(c['file'])
            if is_leak is None:
                raise RuntimeError(f'no leakage verdict for {c["file"]}')
            if not is_leak:
                final_idx.append(i0)
                continue
            # deterministic replacement search: +1,-1,+2,-2,... clamped, skip
            # already-selected or leaking indices
            chosen = None
            off = 1
            while off <= n:
                for cand in (i0 + off, i0 - off):
                    if 0 <= cand < n and cand not in selected_set:
                        chosen = cand
                        break
                if chosen is not None:
                    break
                off += 1
            if chosen is None:
                raise RuntimeError(f'no replacement found for {c["file"]}')
            selected_set.add(chosen)
            final_idx.append(chosen)
            replacements.append({'sequence': seq, 'leaking_index_0based': i0,
                                 'replacement_index_0based': chosen})

        final_idx.sort()
        final_sequences[seq] = {
            'source_video': info['source_video'],
            'n_frames_total': n,
            'selected_frame_indices_0based': final_idx,
            'selected_frame_numbers_1based': [i + 1 for i in final_idx],
        }

    result = {
        'benchmark': 'EyeCU-TEST-v1 FINAL frame list (post-leakage-screen)',
        'replacement_rule': {
            'description': 'nearest ordinal frame in the same sequence, search order +1,-1,+2,-2,+3,-3,... clamped to [0, N-1], skipping frames already selected or themselves leaking',
            'source': 'fixed by the M4 milestone specification, not chosen by this script',
            'content_dependence': 'depends only on the frozen leak/no-leak verdict and frame index arithmetic, never on pixel content directly',
        },
        'leakage_screen_summary': {
            'n_candidates': LEAKAGE['n_candidates'],
            'n_leaks_found': LEAKAGE['n_leaks_found'],
            'result_file': 'LEAKAGE_SCREEN_RESULT.json',
            'result_sha256': hashlib.sha256(
                Path('experiments/records/experiment_M4/LEAKAGE_SCREEN_RESULT.json').read_bytes()
            ).hexdigest(),
        },
        'replacements_applied': replacements,
        'n_replacements_applied': len(replacements),
        'identical_to_initial_list': len(replacements) == 0,
        'sequences': final_sequences,
    }

    out = Path('experiments/records/experiment_M4/FINAL_TEST_FRAME_LIST.json')
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    Path('experiments/records/experiment_M4/FINAL_TEST_FRAME_LIST.sha256').write_text(
        sha + '\n', encoding='utf-8')
    print('n_replacements_applied:', len(replacements))
    print('sha256:', sha)
    print('written:', out)


if __name__ == '__main__':
    main()
