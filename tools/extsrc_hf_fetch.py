#!/usr/bin/env python
"""
Fetch public Hugging Face dataset files into an explicit local destination.

Deliberately not `snapshot_download` into a shared cache: the point is a
reproducible workspace where every file has a recorded revision and hash and
lives at a path we chose. Blobs in ~/.cache are not evidence.

Three things this checks that a plain download does not:

  * the repo revision is pinned and recorded, so "the dataset" means one commit
  * every downloaded file is hashed, and its size is compared to what the API
    said it should be
  * the bytes are inspected for HTML. A 200 response carrying an error page is
    the classic silent failure here, and a 75 MB "zip" that is really an error
    page would otherwise be discovered much later.

No token is read, stored or printed. Public datasets only; a gated repo is
reported as BLOCKED_ACCESS and skipped rather than worked around.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / 'EyeCU_external_data'
BUNDLE = EXT / 'ca_bundle_system.pem'


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def looks_like_html(p: Path) -> bool:
    with open(p, 'rb') as f:
        head = f.read(512).lstrip().lower()
    return head.startswith(b'<!doctype html') or head.startswith(b'<html')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--dest', required=True, help='relative to EyeCU_external_data')
    ap.add_argument('--include', nargs='*', default=None,
                    help='exact paths within the repo; omit for everything')
    ap.add_argument('--max-bytes', type=int, default=0,
                    help='refuse to start if the selection exceeds this')
    ap.add_argument('--log', default=None)
    args = ap.parse_args()

    if BUNDLE.exists():
        os.environ.setdefault('REQUESTS_CA_BUNDLE', str(BUNDLE))
        os.environ.setdefault('CURL_CA_BUNDLE', str(BUNDLE))
    import requests

    api = f'https://huggingface.co/api/datasets/{args.repo}'
    info = requests.get(api, timeout=60)
    if info.status_code in (401, 403):
        print(f'BLOCKED_ACCESS: {args.repo} returned {info.status_code}')
        rec = {'repo': args.repo, 'status': 'BLOCKED_ACCESS',
               'http_status': info.status_code}
        print(json.dumps(rec, indent=1))
        return rec
    info.raise_for_status()
    meta = info.json()
    revision = meta['sha']

    tree, cursor = [], None
    while True:
        r = requests.get(f'{api}/tree/{revision}',
                         params={'recursive': 'true', 'expand': 'true',
                                 **({'cursor': cursor} if cursor else {})}, timeout=120)
        r.raise_for_status()
        tree += r.json()
        link = r.headers.get('link', '')
        if 'rel="next"' in link:
            cursor = link.split('cursor=')[1].split('&')[0].split('>')[0]
        else:
            break
    files = [x for x in tree if x['type'] == 'file']
    if args.include:
        want = set(args.include)
        files = [x for x in files if x['path'] in want]
        missing = want - {x['path'] for x in files}
        if missing:
            print(f'requested but not in the repo: {sorted(missing)}')
    total = sum((x.get('size') or 0) for x in files)
    print(f'{args.repo} @ {revision[:12]}  {len(files)} files  {total/1e6:.1f} MB')
    if args.max_bytes and total > args.max_bytes:
        sys.exit(f'REFUSED: {total/1e6:.1f} MB exceeds the {args.max_bytes/1e6:.1f} MB gate')

    dest = EXT / args.dest
    dest.mkdir(parents=True, exist_ok=True)
    records, t0 = [], time.time()
    for i, x in enumerate(files, 1):
        path = x['path']
        out = dest / path
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f'https://huggingface.co/datasets/{args.repo}/resolve/{revision}/{path}'
        if not out.exists() or (x.get('size') and out.stat().st_size != x['size']):
            with requests.get(url, stream=True, timeout=600) as resp:
                resp.raise_for_status()
                with open(out, 'wb') as f:
                    for chunk in resp.iter_content(1 << 20):
                        f.write(chunk)
        size = out.stat().st_size
        rec = {'path': path, 'bytes': size, 'sha256': sha256(out),
               'expected_bytes': x.get('size'),
               'size_matches_api': (x.get('size') is None or size == x['size']),
               'looks_like_html': looks_like_html(out)}
        if rec['looks_like_html']:
            print(f'  !! {path} looks like an HTML error page, not data')
        if not rec['size_matches_api']:
            print(f'  !! {path} size {size} != API {x.get("size")}')
        records.append(rec)
        if i % 200 == 0 or i == len(files):
            print(f'  {i}/{len(files)}  {time.time()-t0:.0f}s')

    manifest = {
        'repo_id': args.repo,
        'repo_type': 'dataset',
        'revision': revision,
        'downloaded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'license_from_card': (meta.get('cardData') or {}).get('license'),
        'gated': meta.get('gated'),
        'tags': meta.get('tags'),
        'local_path': args.dest,
        'file_count': len(records),
        'total_bytes': sum(r['bytes'] for r in records),
        'all_sizes_match_api': all(r['size_matches_api'] for r in records),
        'any_html_error_pages': any(r['looks_like_html'] for r in records),
        'files': records,
        'credentials_used': 'none -- public dataset, no token read or stored',
    }
    logp = EXT / (args.log or f'huggingface/download_logs/'
                              f'{args.repo.replace("/", "__")}.json')
    logp.parent.mkdir(parents=True, exist_ok=True)
    logp.write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    print(f'\nrevision {revision}')
    print(f'sizes match API: {manifest["all_sizes_match_api"]}  '
          f'html error pages: {manifest["any_html_error_pages"]}')
    print(f'wrote {logp.relative_to(EXT)}')
    return manifest


if __name__ == '__main__':
    main()
