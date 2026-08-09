"""
Cache keys for detection/tracking artifacts.

A cache file is only valid for the exact run that produced it. Reusing one
across a different video, model, detector setting, tracker setting or
skip_frames silently returns results that do not match the current request, so
every one of those inputs is folded into the key (docs/archive/TODO_legacy.md section 6).
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# How much of a video file to read when fingerprinting it. Hashing a full match
# would cost more than the detection run it protects, so identity is taken from
# size + mtime + head/tail sample. That catches replaced or re-encoded files;
# it will not catch an edit in the middle of a file that preserves size and
# mtime, which is not a case the pipeline can produce.
_SAMPLE_BYTES = 1 << 20  # 1 MiB from the start and 1 MiB from the end


def file_fingerprint(path: str, sample_bytes: int = _SAMPLE_BYTES) -> str:
    """Cheap, stable identity for a (possibly large) file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    stat = p.stat()
    h = hashlib.sha256()
    h.update(str(stat.st_size).encode())
    h.update(str(int(stat.st_mtime)).encode())

    with open(p, 'rb') as f:
        h.update(f.read(sample_bytes))
        if stat.st_size > sample_bytes * 2:
            f.seek(-sample_bytes, os.SEEK_END)
            h.update(f.read(sample_bytes))

    return h.hexdigest()


def model_fingerprint(model_path: Optional[str]) -> str:
    """Identity of the detector weights. Falls back to the name if not a file."""
    if not model_path:
        return 'none'
    try:
        return file_fingerprint(model_path)
    except FileNotFoundError:
        # Ultralytics resolves bare names like 'yolov8n.pt' by downloading them;
        # before that happens there is no local file to hash.
        return f'name:{model_path}'


def compute_cache_key(video_path: str,
                      model_path: Optional[str],
                      detector_settings: Dict[str, Any],
                      tracker_settings: Dict[str, Any],
                      skip_frames: int,
                      max_frames: Optional[int] = None,
                      extra: Optional[Dict[str, Any]] = None) -> str:
    """
    Return a short hex key identifying this exact run configuration.

    Any change to the video, the model, the detector settings, the tracker
    settings, skip_frames or max_frames produces a different key, so an
    incompatible cache file can never be picked up.
    """
    payload = {
        'video': file_fingerprint(video_path),
        'model': model_fingerprint(model_path),
        'detector': _stable(detector_settings),
        'tracker': _stable(tracker_settings),
        'skip_frames': int(skip_frames),
        'max_frames': None if max_frames is None else int(max_frames),
        'extra': _stable(extra or {}),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _stable(d: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a settings dict so equal settings always hash equally."""
    return {str(k): (round(v, 6) if isinstance(v, float) else v)
            for k, v in sorted(d.items())}


def cache_path_for(cache_dir, name: str, key: str) -> str:
    """`cache_dir/<name>_<key>.pkl` -- the key is part of the filename so that
    caches for different configurations coexist instead of overwriting."""
    return str(Path(cache_dir) / f'{name}_{key}.pkl')
