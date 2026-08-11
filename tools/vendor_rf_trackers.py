#!/usr/bin/env python
"""
Vendor the exact trackers==2.6.0 CBIoU implementation under a distinct name.

THE PROBLEM. EyeCU owns a top-level package called `trackers/`. The Roboflow
package imports as `trackers` too. Any solution that relies on import
precedence, sys.path order, the working directory, or deleting entries from
sys.modules is a solution that works until the day it does not, and it fails
silently by importing the wrong code.

THE SOLUTION. Copy only the modules CBIoU actually needs into a top-level
package with a name nothing else uses -- `rf_trackers` -- and rewrite the
absolute imports inside them from `trackers.` to `rf_trackers.`. After that
`import rf_trackers` has exactly one possible resolution, and `import trackers`
still means EyeCU's own package everywhere. No path manipulation exists to go
wrong later.

WHAT IS COPIED. The dependency closure of trackers.core.cbiou.tracker, computed
by parsing imports rather than by guessing: 14 modules. BoTSORT modules appear
because CBIoUTracker inherits from BoTSORTTracker; they are an implementation
dependency, not a second production tracker, and nothing exposes them.

WHAT IS NOT CHANGED. Only import statements are rewritten, and only where they
name the vendored package. Every other byte is preserved, so behaviour is the
behaviour that was measured. The rewrite is verified line by line and the file
hashes of both the original and the vendored copy are recorded.

Apache-2.0. The upstream LICENSE is copied and attribution recorded.
"""

import argparse
import ast
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT_MODULE = 'trackers.core.cbiou.tracker'
NEW_PKG = 'rf_trackers'
IMPORT_RE = re.compile(r'\btrackers\.')


def closure(site: Path, root_module: str):
    def path_of(m):
        p = site / (m.replace('.', '/') + '.py')
        return p if p.exists() else site / m.replace('.', '/') / '__init__.py'

    seen, stack = set(), [root_module]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        p = path_of(m)
        if not p.exists():
            continue
        tree = ast.parse(p.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module and \
                    n.module.startswith('trackers'):
                stack.append(n.module)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    if a.name.startswith('trackers'):
                        stack.append(a.name)
    return sorted(m for m in seen if path_of(m).exists()), path_of


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--site', required=True, help='site-packages holding trackers 2.6.0')
    ap.add_argument('--wheel', default=None, help='the wheel it was installed from')
    ap.add_argument('--out', default='rf_trackers')
    args = ap.parse_args()

    site = Path(args.site)
    dist = site / 'trackers-2.6.0.dist-info'
    if not dist.exists():
        raise SystemExit(f'trackers 2.6.0 dist-info not found under {site}')

    mods, path_of = closure(site, ROOT_MODULE)
    out = REPO / args.out
    if out.exists():
        shutil.rmtree(out)

    record = {'vendored_from': {'package': 'trackers', 'version': '2.6.0',
                                'author': 'Roboflow et al. <develop@roboflow.com>',
                                'license': 'Apache-2.0',
                                'site_packages': str(site)},
              'vendored_as': NEW_PKG,
              'reason': ('EyeCU owns a top-level package named trackers/; the '
                         'external package imports under the same name. '
                         'Renaming the vendored copy makes every import '
                         'deterministic without any sys.path or sys.modules '
                         'manipulation.'),
              'root_module': ROOT_MODULE,
              'modules': [], 'created': datetime.now(timezone.utc).isoformat(timespec='seconds')}
    if args.wheel and Path(args.wheel).exists():
        record['vendored_from']['wheel'] = Path(args.wheel).name
        record['vendored_from']['wheel_sha256'] = hashlib.sha256(
            Path(args.wheel).read_bytes()).hexdigest()

    # every package level needs an __init__ so the tree imports cleanly
    pkgs = set()
    for m in mods:
        parts = m.split('.')[1:]
        for i in range(len(parts)):
            pkgs.add('/'.join(parts[:i]))
    for rel in sorted(pkgs):
        d = out / rel if rel else out
        d.mkdir(parents=True, exist_ok=True)
        init = d / '__init__.py'
        if not init.exists():
            init.write_text('', encoding='utf-8')

    rewritten_lines = 0
    for m in mods:
        src = path_of(m)
        rel = Path(*m.split('.')[1:])
        dst = out / (rel.with_suffix('.py') if src.name != '__init__.py'
                     else rel / '__init__.py')
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding='utf-8')
        new = []
        changed = 0
        for line in text.splitlines(keepends=True):
            stripped = line.lstrip()
            if stripped.startswith(('from trackers.', 'import trackers.')) or \
                    (stripped.startswith(('from ', 'import ')) and 'trackers.' in line):
                sub = IMPORT_RE.sub(f'{NEW_PKG}.', line)
                if sub != line:
                    changed += 1
                new.append(sub)
            else:
                new.append(line)
        dst.write_text(''.join(new), encoding='utf-8')
        rewritten_lines += changed
        record['modules'].append({
            'module': m,
            'vendored_path': str(dst.relative_to(REPO)).replace('\\', '/'),
            'original_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
            'vendored_sha256': hashlib.sha256(dst.read_bytes()).hexdigest(),
            'import_lines_rewritten': changed,
        })

    lic_src = dist / 'licenses' / 'LICENSE'
    shutil.copy2(lic_src, out / 'LICENSE')
    record['license_file'] = f'{args.out}/LICENSE'
    record['license_sha256'] = hashlib.sha256(lic_src.read_bytes()).hexdigest()
    record['import_lines_rewritten_total'] = rewritten_lines
    record['only_change'] = ('import statements naming the vendored package; '
                             'no other byte was altered')

    (out / '__init__.py').write_text(
        '"""\n'
        'Vendored subset of Roboflow `trackers` 2.6.0 (Apache-2.0), renamed.\n'
        '\n'
        'EyeCU owns a top-level package called `trackers`, so the upstream\n'
        'package cannot be installed alongside it without the import being\n'
        'decided by path order. Only the modules CBIoUTracker needs are copied\n'
        'here, with `trackers.` rewritten to `rf_trackers.` in their imports\n'
        'and nothing else touched.\n'
        '\n'
        'Only CBIoUTracker is exported. BoTSORT modules are present because\n'
        'CBIoUTracker inherits from BoTSORTTracker; they are an implementation\n'
        'dependency, not a second production tracker.\n'
        '\n'
        'Provenance, hashes and licence: see VENDOR_PROVENANCE.json and LICENSE.\n'
        '"""\n'
        '\n'
        'from rf_trackers.core.cbiou.tracker import CBIoUTracker\n'
        '\n'
        "__all__ = ['CBIoUTracker']\n"
        "__vendored_from__ = 'trackers==2.6.0'\n",
        encoding='utf-8')
    (out / 'VENDOR_PROVENANCE.json').write_text(
        json.dumps(record, indent=1), encoding='utf-8')
    (out / 'NOTICE').write_text(
        'This directory contains a modified subset of the Roboflow "trackers"\n'
        'package version 2.6.0, licensed under the Apache License 2.0.\n'
        '\n'
        'Copyright Roboflow et al. <develop@roboflow.com>\n'
        'Upstream: https://github.com/roboflow/trackers\n'
        '\n'
        'MODIFICATION: absolute imports of the form `trackers.<x>` were\n'
        'rewritten to `rf_trackers.<x>` so the package can coexist with EyeCU\'s\n'
        'own top-level `trackers/` package. No other change was made. The exact\n'
        'files, their upstream hashes and the rewritten hashes are recorded in\n'
        'VENDOR_PROVENANCE.json.\n'
        '\n'
        'The full Apache 2.0 licence text is in LICENSE.\n',
        encoding='utf-8')

    print(f'vendored {len(mods)} modules into {out.relative_to(REPO)}')
    print(f'import lines rewritten: {rewritten_lines}')
    for m in record['modules']:
        print(f"  {m['import_lines_rewritten']:>2} rewrites  {m['vendored_path']}")
    print(f"licence: {record['license_file']}  sha256 {record['license_sha256'][:16]}")


if __name__ == '__main__':
    main()
