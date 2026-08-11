#!/usr/bin/env python
"""
Parallel variant of extsrc_hf_fetch for large public payloads.

The sequential fetcher managed 122 KB/s on the 7 GB SoccerNet-V3 export -- not
because the link is slow but because every one of 4,797 files paid a fresh TLS
handshake through an intercepting proxy. A pooled session plus a modest thread
count removes that cost.

Same guarantees as the sequential version, which is the point of having both:
the revision is pinned, every file is hashed, sizes are checked against the API,
and the first bytes are inspected so an HTML error page cannot masquerade as a
PNG. Resumable -- a file already present at the right size is not refetched.
"""

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
BUNDLE = EXT / 'ca_bundle_system.pem'
_local = threading.local()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def session():
    import requests
    if not hasattr(_local, 's'):
        s = requests.Session()
        a = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8,
                                          max_retries=3)
        s.mount('https://', a)
        _local.s = s
    return _local.s


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dest', required=True)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--max-bytes', type=int, default=0)
    ap.add_argument('--log', required=True)
    args = ap.parse_args()

    if BUNDLE.exists():
        os.environ.setdefault('REQUESTS_CA_BUNDLE', str(BUNDLE))
        os.environ.setdefault('CURL_CA_BUNDLE', str(BUNDLE))
    import requests

    api = f'https://huggingface.co/api/datasets/{args.repo}'
    meta = session().get(api, timeout=60)
    if meta.status_code in (401, 403):
        print(f'BLOCKED_ACCESS: {args.repo} -> {meta.status_code}')
        return
    meta.raise_for_status()
    meta = meta.json()
    revision = meta['sha']

    tree, cursor = [], None
    while True:
        r = session().get(f'{api}/tree/{revision}',
                          params={'recursive': 'true', 'expand': 'true',
                                  **({'cursor': cursor} if cursor else {})},
                          timeout=120)
        r.raise_for_status()
        tree += r.json()
        link = r.headers.get('link', '')
        if 'rel="next"' in link:
            cursor = link.split('cursor=')[1].split('&')[0].split('>')[0]
        else:
            break
    files = [x for x in tree if x['type'] == 'file']
    total = sum((x.get('size') or 0) for x in files)
    print(f'{args.repo} @ {revision[:12]}  {len(files)} files  {total/1e9:.2f} GB')
    if args.max_bytes and total > args.max_bytes:
        sys.exit(f'REFUSED: exceeds gate')

    dest = EXT / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def fetch(x):
        path = x['path']
        out = dest / path
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists() or (x.get('size') and out.stat().st_size != x['size']):
            url = (f'https://huggingface.co/datasets/{args.repo}/resolve/'
                   f'{revision}/{path}')
            with session().get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                tmp = out.with_suffix(out.suffix + '.part')
                with open(tmp, 'wb') as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(out)
        with open(out, 'rb') as f:
            head = f.read(64).lstrip().lower()
        rec = {'path': path, 'bytes': out.stat().st_size,
               'sha256': sha256(out), 'expected_bytes': x.get('size'),
               'size_matches_api': (x.get('size') is None
                                    or out.stat().st_size == x['size']),
               'looks_like_html': head.startswith(b'<!doctype html')
                                  or head.startswith(b'<html')}
        with lock:
            done[0] += 1
            if done[0] % 400 == 0 or done[0] == len(files):
                el = time.time() - t0
                print(f'  {done[0]}/{len(files)}  {el:.0f}s  '
                      f'{sum(1 for _ in [1]) and done[0]/max(el,1):.1f} files/s')
        return rec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(fetch, files))

    manifest = {
        'repo_id': args.repo, 'repo_type': 'dataset', 'revision': revision,
        'downloaded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'license_from_card': (meta.get('cardData') or {}).get('license'),
        'gated': meta.get('gated'), 'local_path': args.dest,
        'file_count': len(records),
        'total_bytes': sum(r['bytes'] for r in records),
        'all_sizes_match_api': all(r['size_matches_api'] for r in records),
        'any_html_error_pages': any(r['looks_like_html'] for r in records),
        'files': records,
        'credentials_used': 'none -- public dataset, no token read or stored',
    }
    logp = EXT / args.log
    logp.parent.mkdir(parents=True, exist_ok=True)
    logp.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(f'\nsizes match API: {manifest["all_sizes_match_api"]}  '
          f'html pages: {manifest["any_html_error_pages"]}  '
          f'{manifest["total_bytes"]/1e9:.2f} GB in {time.time()-t0:.0f}s')
    print(f'wrote {args.log}')


if __name__ == '__main__':
    main()
