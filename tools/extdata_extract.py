#!/usr/bin/env python
"""
Stage 0/1 of the external-data audit: hash the ZIPs, extract into isolation.

The ZIPs are evidence, not inputs. They are hashed before anything is read out
of them and re-hashed at the end of the audit; nothing here opens them for
writing, and nothing is extracted anywhere near data/.

Roboflow exports carry their own provenance -- workspace, project, version,
licence, and crucially the pre-processing and augmentation that were applied
before export. That last part decides how the rest of the audit must read the
numbers, so it is parsed here rather than eyeballed later.
"""

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ZIPS = REPO / 'check_datasets'
AUDIT = REPO / 'experiments' / 'external_data_audit'

# Stable short ids. Three ZIPs are literally named "soccer" by three unrelated
# Roboflow users, so filenames cannot identify a source; workspace/project can.
SOURCES = {
    'S1': 'SOccer.v1i.yolo26.zip',
    'S2': 'football.v1i.yolo26.zip',
    'S3': 'soccer players.v1-release-640.yolo26.zip',
    'S4': 'soccer.v1i.yolo26 (1).zip',
    'S5': 'soccer.v1i.yolo26 (2).zip',
    'S6': 'soccer.v1i.yolo26 (3).zip',
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def parse_readme(text: str) -> dict:
    """Pull the export facts out of README.roboflow.txt.

    The pre-processing line is the one that matters most: five of six exports
    were resized by Roboflow before export, so a ball measured in the stored
    image is not a ball measured in the original frame.
    """
    out = {'declared_images': None, 'preprocessing': [], 'augmentation': [],
           'augmentation_multiplier': None, 'export_date': None}
    m = re.search(r'The dataset includes (\d+) images', text)
    if m:
        out['declared_images'] = int(m.group(1))
    m = re.search(r'exported via roboflow\.com on ([^\n]+)', text)
    if m:
        out['export_date'] = m.group(1).strip()
    m = re.search(r'augmentation was applied to create (\d+) versions', text)
    if m:
        out['augmentation_multiplier'] = int(m.group(1))
    sec = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('The following pre-processing'):
            sec = 'preprocessing'
            continue
        if s.startswith('The following augmentation'):
            sec = 'augmentation'
            continue
        if s.startswith('No image augmentation'):
            sec = None
            out['augmentation_multiplier'] = 1
            continue
        if s.startswith('*') and sec:
            out[sec].append(s.lstrip('* ').strip())
        elif not s:
            sec = sec if sec and not out[sec] else None
    return out


def main():
    import yaml
    AUDIT.mkdir(parents=True, exist_ok=True)
    for d in ('raw', 'extracted', 'reports', 'contact_sheets', 'candidate_index'):
        (AUDIT / d).mkdir(exist_ok=True)

    record = {'audit': 'EyeCU external data audit', 'zip_dir': str(ZIPS),
              'note': ('ZIPs are left in place and treated as immutable evidence; '
                       'hashes below are re-verified at the end of the audit'),
              'sources': {}}

    for sid, fname in SOURCES.items():
        zp = ZIPS / fname
        if not zp.exists():
            sys.exit(f'missing ZIP: {fname}')
        digest = sha256(zp)
        z = zipfile.ZipFile(zp)
        names = z.namelist()
        y = yaml.safe_load(z.read('data.yaml').decode('utf-8'))
        rf = y.get('roboflow', {}) or {}
        readme = parse_readme(z.read('README.roboflow.txt').decode('utf-8', 'replace'))

        dst = AUDIT / 'extracted' / sid
        if not dst.exists():
            dst.mkdir(parents=True)
            z.extractall(dst)
        n_img = sum(1 for n in names if '/images/' in n and not n.endswith('/'))
        n_lbl = sum(1 for n in names if '/labels/' in n and not n.endswith('/'))

        record['sources'][sid] = {
            'original_filename': fname,
            'sha256': digest,
            'archive_bytes': zp.stat().st_size,
            'workspace': rf.get('workspace'),
            'project': rf.get('project'),
            'version': rf.get('version'),
            'roboflow_url': rf.get('url'),
            'license': rf.get('license'),
            'export_format': 'YOLO26 (Ultralytics YOLO txt + data.yaml)',
            'declared_classes': y.get('names'),
            'declared_nc': y.get('nc'),
            'declared_image_count': readme['declared_images'],
            'export_date': readme['export_date'],
            'preprocessing': readme['preprocessing'],
            'augmentation': readme['augmentation'],
            'augmentation_multiplier': readme['augmentation_multiplier'],
            'zip_entries': len(names),
            'zip_image_files': n_img,
            'zip_label_files': n_lbl,
            'splits_present': sorted({n.split('/')[0] for n in names
                                      if '/' in n and n.split('/')[0] in
                                      ('train', 'valid', 'test')}),
            'extracted_to': str(dst.relative_to(REPO)).replace('\\', '/'),
        }
        print(f'{sid}  {rf.get("workspace")}/{rf.get("project")} v{rf.get("version")}  '
              f'{n_img:>5} images  {readme["augmentation_multiplier"]}x aug  '
              f'{rf.get("license")}')

    (AUDIT / 'raw' / 'SOURCES.json').write_text(
        json.dumps(record, indent=1, ensure_ascii=False), encoding='utf-8')
    print(f'\nwrote {(AUDIT / "raw" / "SOURCES.json").relative_to(REPO)}')


if __name__ == '__main__':
    main()
