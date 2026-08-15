#!/usr/bin/env python
"""
One deterministic way to turn a package IMAGE value into a file on disk.

There were two copies of this logic. They differed by one character -- `extract`
against `extracted` -- and the second server therefore served a 404 for every
image while its queue, its state and its whole UI loaded perfectly. The reviewer
saw a broken-image icon and a working panel, which is the worst possible split:
everything says the tool is fine except the one thing being reviewed.

Both copies also searched with rglob, so a file was found by BASENAME anywhere
below the split directory. That works until two files share a name, and then it
silently returns whichever the filesystem happened to walk first. Image identity
is metadata, not a search result: IMAGE is `<split>/<file>` exactly as recorded
in the ledger, and it resolves to exactly one path or to nothing.

    resolve('test/515_pp_jpg.rf.c640....jpg')
      -> EyeCU_external_data/.../extracted/test/515_pp_jpg.rf.c640....jpg

The immutable extract is read-only here. Nothing in this module writes.
"""

import mimetypes
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
IMGROOT = (REPO / 'EyeCU_external_data' / 'huggingface'
           / 'keremberke_football_object_detection' / 'extracted')
SPLITS = ('train', 'valid', 'test')


class ImageError(Exception):
    """Why an IMAGE could not be served. Always says which one, and why."""


def normalise(image: str) -> str:
    """`train\\x.jpg` and `train/x.jpg` are the same image. Traversal is not.

    The value comes from the ledger, but it also arrives over HTTP, so it is
    treated as untrusted: a `..` segment would let a request walk out of the
    immutable extract entirely.
    """
    s = str(image or '').replace('\\', '/').strip().lstrip('/')
    if not s:
        raise ImageError('empty IMAGE value')
    parts = [p for p in s.split('/') if p not in ('', '.')]
    if any(p == '..' for p in parts):
        raise ImageError(f'IMAGE must not contain a parent segment: {image!r}')
    if len(parts) < 2:
        raise ImageError(f'IMAGE must be "<split>/<file>", got {image!r}')
    if parts[0] not in SPLITS:
        raise ImageError(f'unknown split {parts[0]!r} in {image!r}; '
                         f'expected one of {SPLITS}')
    return '/'.join(parts)


def path_for(image: str, root: Path = None) -> Path:
    """The one path this IMAGE means. No search, no basename fallback."""
    return (Path(root) if root else IMGROOT).joinpath(*normalise(image).split('/'))


def resolve(image: str, root: Path = None) -> Path:
    """The file, or ImageError naming the exact path that was looked for."""
    p = path_for(image, root)
    if not p.is_file():
        raise ImageError(f'no file at {p}')
    return p


def read(image: str, root: Path = None):
    """(bytes, content-type). Bytes are read with pathlib, so a non-ASCII
    parent directory is handled by the OS rather than by an encoder -- the
    project lives under a Hebrew path and cv2.imread already failed on it once."""
    p = resolve(image, root)
    return p.read_bytes(), mimetypes.guess_type(p.name)[0] or 'image/jpeg'


def preflight(images, root: Path = None):
    """Check a whole queue before a human starts. Returns (ok, problems).

    A tool that opens a browser and then 404s on the first image has wasted the
    reviewer's setup and told them nothing useful. Checking up front turns that
    into one clear message before anything opens.
    """
    problems = []
    for im in sorted(set(images)):
        try:
            resolve(im, root)
        except ImageError as e:
            problems.append((im, str(e)))
    return (not problems), problems
