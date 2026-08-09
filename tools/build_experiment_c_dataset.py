"""Assemble the Experiment C training dataset: original TRAIN + derived, frozen VAL."""
import hashlib, json, shutil, sys, subprocess, zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, '.')
from tools.build_derived_train import VAL_MATCHES, TEST_MATCHES

SRC = Path('data/dataset_baseline')
DER = Path('data/derived_train')
OUT = Path('data/dataset_C')

if OUT.exists():
    shutil.rmtree(OUT)
for sp in ('train', 'val'):
    (OUT / 'images' / sp).mkdir(parents=True)
    (OUT / 'labels' / sp).mkdir(parents=True)

def copy(src_img_dir, src_lbl_dir, split, files=None):
    n = 0
    for ip in sorted(src_img_dir.glob('*.jpg')):
        if files is not None and ip.name not in files:
            continue
        shutil.copy2(ip, OUT / 'images' / split / ip.name)
        lp = src_lbl_dir / f'{ip.stem}.txt'
        dst = OUT / 'labels' / split / f'{ip.stem}.txt'
        if lp.exists():
            shutil.copy2(lp, dst)
        else:
            dst.write_text('', encoding='utf-8')
        n += 1
    return n

n_orig = copy(SRC/'images/train', SRC/'labels/train', 'train')
n_der  = copy(DER/'images', DER/'labels', 'train')
n_val  = copy(SRC/'images/val', SRC/'labels/val', 'val')
print(f'train: {n_orig} original + {n_der} derived = {n_orig+n_der}')
print(f'val:   {n_val} (frozen, unchanged)')

# ---- leakage assertions BEFORE writing the yaml
train_files = {p.name for p in (OUT/'images/train').glob('*.jpg')}
val_files   = {p.name for p in (OUT/'images/val').glob('*.jpg')}
tv_files    = {p.name for p in Path('data/temporal_val/images').glob('*.jpg')}
assert not (train_files & val_files), 'train/val file overlap'
assert not (train_files & tv_files), 'temporal VAL leaked into train'
assert not (val_files & tv_files), 'temporal VAL leaked into val'
train_matches = {p.stem.rsplit('_', 1)[0].replace('_dpos', '_').replace('_dneg', '_')
                 for p in (OUT/'images/train').glob('*.jpg')}
der_sources = {i['source_match'] for i in
               json.loads((DER/'manifest.json').read_text(encoding='utf-8'))['items']}
assert not (der_sources & (VAL_MATCHES | TEST_MATCHES)), 'derived from held-out source'
val_matches = {p.stem.rsplit('_', 1)[0] for p in (OUT/'images/val').glob('*.jpg')}
assert val_matches == VAL_MATCHES, f'val matches changed: {val_matches}'
print('leakage assertions: PASS')

# ---- class histogram
hist = Counter()
for lp in (OUT/'labels/train').glob('*.txt'):
    for line in lp.read_text(encoding='utf-8').splitlines():
        if line.strip():
            hist[int(line.split()[0])] += 1
names = ['player', 'goalkeeper', 'referee', 'ball']
print('train instances:', {names[k]: v for k, v in sorted(hist.items())})

yaml = (
    "# EyeCU Experiment C -- original TRAIN + derived scale/context + hard negatives\n"
    "# VAL is the frozen 208-image set, unchanged from Experiment A.\n"
    "path: /content/football_dataset_C\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n  0: player\n  1: goalkeeper\n  2: referee\n  3: ball\n"
)
(OUT / 'football_C.yaml').write_text(yaml, encoding='utf-8')

def sha(p):
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()[:16]

def dirsha(d):
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob('*')):
        if p.is_file():
            h.update(p.name.encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]

git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
repro = {
    'experiment': 'C_yolo26s_960_scale_context_hardneg',
    'git_commit': git,
    'seed': 0,
    'model_start': 'yolo26s.pt (COCO pretrained, same as A)',
    'imgsz': 960,
    'n_train_original': n_orig, 'n_train_derived': n_der,
    'n_train_total': n_orig + n_der, 'n_val': n_val,
    'derived_manifest_sha256_16': sha(DER/'manifest.json'),
    'derived_images_dir_sha256_16': dirsha(DER/'images'),
    'derived_labels_dir_sha256_16': dirsha(DER/'labels'),
    'val_images_dir_sha256_16': dirsha(SRC/'images/val'),
    'val_labels_dir_sha256_16': dirsha(SRC/'labels/val'),
    'yaml_sha256_16': sha(OUT/'football_C.yaml'),
    'train_class_instances': {names[k]: v for k, v in sorted(hist.items())},
}
(OUT / 'reproducibility.json').write_text(json.dumps(repro, indent=2), encoding='utf-8')
print(json.dumps(repro, indent=2))

zp = Path('data/football_dataset_C.zip')
with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(OUT).as_posix())
print(f'\nzip: {zp}  ({zp.stat().st_size/1e6:.1f} MB)')
