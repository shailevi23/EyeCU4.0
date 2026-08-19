"""Build the three Stage-A arms (A00 / A10 / A11) as independent YOLO datasets.

Stage-A is a controlled label-set ablation. All three arms hold the identical
1,030 training images -- 823 EyeCU TRAIN plus the 207 Round-0 reviewed images
from the keremberke source TRAIN split -- and differ only in annotation content:

    A00   EyeCU labels + source-TRAIN observed labels (semantic class remap)
    A10   A00 + 22 effective ACTIVE_MATCH_BALL reviewed additions
    A11   A10 + 74 effective NON_ACTIVE_EXTRA_BALL reviewed additions

The governing design is STAGE_A_DESIGN_V2.json; the binding ontology is
BALL_ONTOLOGY_POLICY.json. Both are read here and asserted against, so a drift
in either one fails the build rather than silently producing a different
experiment.

This writes only to --out. It never touches the baseline dataset, the source
export, the decision log, or any frozen artifact. EyeCU sealed TEST is never
read: the baseline split contributes train/ and val/ only, and no code path
below can name a test directory.

Usage:
    python tools/build_stage_a_dataset.py --out <new-empty-directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# EyeCU detector schema, from data/dataset_baseline/football.yaml.
EYECU_NAMES = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}

# The source export names its classes differently and orders them differently.
# The mapping is established by NAME, never by trusting the numeric ids: the
# two schemas happen to use overlapping integers for different things, so an
# id-based merge would silently relabel every box in the dataset.
SOURCE_TO_EYECU_NAME = {
    "football": "ball",
    "player": "player",
    "goalkeeper": "goalkeeper",
    "referee": "referee",
}

ACTIVE = "ACTIVE_MATCH_BALL"
NON_ACTIVE = "NON_ACTIVE_EXTRA_BALL"

REPO = Path(__file__).resolve().parents[1]
KB_REVIEW = REPO / "experiments" / "external_sources" / "keremberke_review"


class BuildError(RuntimeError):
    """A fail-closed build violation. Never caught -- it aborts the build."""


def fail(msg: str) -> None:
    raise BuildError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(paths: list[Path]) -> str:
    """Order-independent digest of a file set: name + content, sorted."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode("utf-8"))
        h.update(sha256_file(p).encode("ascii"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load_design(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    arms = d["arms"]
    add = d["train_restricted_additions"]
    expected = {
        "eyecu_train": arms["eyecu_train"],
        "external_train": arms["external_added"],
        "total": arms["shared_image_set"],
        "active": add["active"],
        "nonactive": add["nonactive"],
        "additions_total": add["total"],
    }
    if expected["eyecu_train"] + expected["external_train"] != expected["total"]:
        fail(
            "design is internally inconsistent: "
            f"{expected['eyecu_train']} + {expected['external_train']} "
            f"!= {expected['total']}"
        )
    if expected["active"] + expected["nonactive"] != expected["additions_total"]:
        fail("design additions do not sum to the declared total")
    return expected


def assert_ontology(path: Path) -> None:
    o = json.loads(path.read_text(encoding="utf-8"))
    got = o.get("BALL_DETECTOR_ONTOLOGY")
    if got != "ALL_VISIBLE_PHYSICAL_FOOTBALLS":
        fail(f"ball ontology is {got!r}, not ALL_VISIBLE_PHYSICAL_FOOTBALLS")
    if o.get("status") != "BINDING":
        fail("ball ontology policy is not marked BINDING")


def round0_train_images(sample_path: Path) -> tuple[list[str], Counter]:
    """The 207 source-TRAIN images drawn in Round 0, and the full split census.

    The draw covers all three source splits. Only TRAIN may reach Stage-A; the
    60 valid and 33 test images are counted here purely so the exclusion is
    positively demonstrated rather than assumed.
    """
    s = json.loads(sample_path.read_text(encoding="utf-8"))
    census = Counter(item["IMAGE"].split("/", 1)[0] for item in s["sample"])
    train = sorted({i["IMAGE"] for i in s["sample"] if i["IMAGE"].startswith("train/")})
    if len(train) != census["train"]:
        fail("duplicate image ids in the Round-0 train slice")
    return train, census


def effective_ball_roles(decisions_path: Path) -> dict[str, dict]:
    """Latest human ball-role decision per Round-0 object.

    The log is append-only and a reviewer may revisit an object, so the last
    row for an object id is the effective one. Ordering is file order, which is
    the chronological record.
    """
    latest: dict[str, dict] = {}
    with open(decisions_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("mode") != "ball_ontology_revisit":
                continue
            latest[row["missing_object_id"]] = row
    return latest


def load_source_export(export_path: Path, wanted: set[str]) -> tuple[dict, dict]:
    """Source COCO restricted to `wanted`, with the class remap applied by name.

    Returns (images_by_filename, annotations_by_filename) where every annotation
    already carries an EyeCU class id.
    """
    d = json.loads(export_path.read_text(encoding="utf-8"))

    names = {c["name"] for c in d["categories"]}
    if names != set(SOURCE_TO_EYECU_NAME):
        fail(
            "source category names are "
            f"{sorted(names)}, expected {sorted(SOURCE_TO_EYECU_NAME)}"
        )
    eyecu_id = {v: k for k, v in EYECU_NAMES.items()}
    remap = {
        c["id"]: eyecu_id[SOURCE_TO_EYECU_NAME[c["name"]]] for c in d["categories"]
    }

    images = {}
    by_id = {}
    for im in d["images"]:
        key = "train/" + im["file_name"]
        if key not in wanted:
            continue
        if not im.get("width") or not im.get("height"):
            fail(f"source image {key} has no recorded dimensions")
        images[key] = im
        by_id[im["id"]] = key

    missing = wanted - set(images)
    if missing:
        fail(f"{len(missing)} Round-0 train images absent from the export: "
             f"{sorted(missing)[:5]}")

    anns: dict[str, list] = {k: [] for k in images}
    for a in sorted(d["annotations"], key=lambda x: x["id"]):
        key = by_id.get(a["image_id"])
        if key is None:
            continue
        anns[key].append({"cls": remap[a["category_id"]], "bbox": a["bbox"]})
    return images, anns


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def coco_to_yolo(bbox, img_w: int, img_h: int, label: str) -> tuple[float, ...]:
    """COCO [x, y, w, h] in source pixels -> YOLO normalized cxcywh.

    Malformed geometry fails the build. It is never clamped: a box that leaves
    the frame means the object was matched to the wrong image or the wrong
    coordinate space, and silently squashing it would hide that.
    """
    x, y, w, h = (float(v) for v in bbox)
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        fail(f"{label}: centre ({cx:.6f}, {cy:.6f}) outside the image")
    if not (0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
        fail(f"{label}: extent ({nw:.6f}, {nh:.6f}) not in (0, 1]")
    return cx, cy, nw, nh


def fmt(cls: int, geom) -> str:
    return f"{cls} " + " ".join(f"{v:.6f}" for v in geom)


# ---------------------------------------------------------------------------
# materialisation
# ---------------------------------------------------------------------------


def place(src: Path, dst: Path) -> None:
    """Hardlink where the filesystem allows it, else copy.

    The three arms share every image byte-for-byte, so linking keeps the build
    at one copy on disk. The fallback keeps it working across filesystems.
    """
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def build_arm(
    arm: str,
    out_root: Path,
    eyecu_root: Path,
    kb_images_root: Path,
    ext_images: dict,
    ext_labels: dict[str, list[str]],
    val_images: list[Path],
) -> dict:
    root = out_root / arm
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    stems: set[str] = set()
    counts = Counter()
    n_eyecu = 0

    # EyeCU TRAIN: images and labels are already in the detector schema and are
    # identical in every arm.
    for img in sorted((eyecu_root / "images" / "train").iterdir()):
        if not img.is_file():
            continue
        stem = img.stem
        if stem in stems:
            fail(f"duplicate training stem {stem!r}")
        stems.add(stem)
        place(img, root / "images/train" / img.name)
        lbl = eyecu_root / "labels" / "train" / f"{stem}.txt"
        if not lbl.exists():
            fail(f"EyeCU train image {img.name} has no label file")
        text = lbl.read_text(encoding="utf-8")
        (root / "labels/train" / f"{stem}.txt").write_text(text, encoding="utf-8")
        for line in text.splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
        n_eyecu += 1

    # External source TRAIN: the 207 reviewed images, labels rebuilt per arm.
    n_ext = 0
    for key in sorted(ext_images):
        src = kb_images_root / key
        if not src.exists():
            fail(f"source image missing on disk: {src}")
        stem = src.stem
        if stem in stems:
            fail(f"external image {key} collides with an EyeCU stem {stem!r}")
        stems.add(stem)
        place(src, root / "images/train" / src.name)
        lines = ext_labels[key]
        (root / "labels/train" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        for line in lines:
            counts[int(line.split()[0])] += 1
        n_ext += 1

    # Validation: the existing EyeCU val split, untouched and shared by all arms.
    n_val = 0
    for img in val_images:
        place(img, root / "images/val" / img.name)
        lbl = eyecu_root / "labels" / "val" / f"{img.stem}.txt"
        if not lbl.exists():
            fail(f"EyeCU val image {img.name} has no label file")
        shutil.copy2(lbl, root / "labels/val" / f"{img.stem}.txt")
        n_val += 1

    yaml = root / "football_stage_a.yaml"
    yaml.write_text(
        f"# Stage-A arm {arm} -- generated by tools/build_stage_a_dataset.py\n"
        f"# Labels differ between arms; the {n_eyecu + n_ext} training images do not.\n"
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(EYECU_NAMES.items())),
        encoding="utf-8",
    )

    return {
        "arm": arm,
        "root": str(root),
        "yaml": str(yaml),
        "train_images": n_eyecu + n_ext,
        "eyecu_train_images": n_eyecu,
        "external_train_images": n_ext,
        "val_images": n_val,
        "total_labels": sum(counts.values()),
        "ball_labels": counts[3],
        "labels_by_class": {EYECU_NAMES[k]: v for k, v in sorted(counts.items())},
        "train_stems_sha256": hashlib.sha256(
            "\n".join(sorted(stems)).encode("utf-8")
        ).hexdigest(),
    }


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path,
                    help="new output directory; must not be an existing dataset")
    ap.add_argument("--eyecu-root", type=Path,
                    default=REPO / "data" / "dataset_baseline")
    ap.add_argument("--source-export", type=Path,
                    default=KB_REVIEW / "repaired_export" / "train_annotations.coco.json")
    ap.add_argument("--source-images", type=Path,
                    default=REPO / "EyeCU_external_data" / "huggingface"
                    / "keremberke_football_object_detection" / "extracted")
    ap.add_argument("--round0-sample", type=Path,
                    default=KB_REVIEW / "BALL_QA_ROUND0_SAMPLE.json")
    ap.add_argument("--decisions", type=Path, default=KB_REVIEW / "decisions.json")
    ap.add_argument("--design", type=Path, default=KB_REVIEW / "STAGE_A_DESIGN_V2.json")
    ap.add_argument("--ontology", type=Path,
                    default=KB_REVIEW / "BALL_ONTOLOGY_POLICY.json")
    args = ap.parse_args()

    for p in (args.eyecu_root, args.source_export, args.source_images,
              args.round0_sample, args.decisions, args.design, args.ontology):
        if not p.exists():
            fail(f"required input missing: {p}")

    out = args.out
    if out.exists() and any(out.iterdir()):
        fail(f"--out {out} is not empty; Stage-A builds to a fresh directory")
    out.mkdir(parents=True, exist_ok=True)

    expected = load_design(args.design)
    assert_ontology(args.ontology)

    # -- population ---------------------------------------------------------
    train_keys, census = round0_train_images(args.round0_sample)
    if len(train_keys) != expected["external_train"]:
        fail(f"Round-0 train slice is {len(train_keys)}, "
             f"design says {expected['external_train']}")
    wanted = set(train_keys)

    leak_valid = sum(1 for k in wanted if k.startswith("valid/"))
    leak_test = sum(1 for k in wanted if k.startswith("test/"))
    if leak_valid or leak_test:
        fail(f"source holdout leaked into TRAIN: valid={leak_valid} test={leak_test}")

    ext_images, ext_observed = load_source_export(args.source_export, wanted)

    # -- reviewed additions -------------------------------------------------
    roles = effective_ball_roles(args.decisions)
    train_objs = {
        oid: r for oid, r in roles.items() if r["IMAGE"] in wanted
    }
    role_census = Counter(r["HUMAN_BALL_ROLE"] for r in train_objs.values())
    unknown = set(role_census) - {ACTIVE, NON_ACTIVE}
    if unknown:
        fail(f"unresolved ball roles inside the Stage-A train slice: {sorted(unknown)}")
    if role_census[ACTIVE] != expected["active"]:
        fail(f"ACTIVE additions are {role_census[ACTIVE]}, "
             f"design says {expected['active']}")
    if role_census[NON_ACTIVE] != expected["nonactive"]:
        fail(f"NON_ACTIVE additions are {role_census[NON_ACTIVE]}, "
             f"design says {expected['nonactive']}")

    ball_cls = 3
    additions: dict[str, dict[str, list[str]]] = {
        ACTIVE: {k: [] for k in wanted},
        NON_ACTIVE: {k: [] for k in wanted},
    }
    for oid in sorted(train_objs):
        row = train_objs[oid]
        key = row["IMAGE"]
        bbox = row.get("round0_bbox_xywh")
        if not bbox or len(bbox) != 4:
            fail(f"object {oid} on {key} has no usable Round-0 geometry")
        im = ext_images[key]
        geom = coco_to_yolo(bbox, im["width"], im["height"], f"{oid} on {key}")
        additions[row["HUMAN_BALL_ROLE"]][key].append(fmt(ball_cls, geom))

    # -- per-arm label sets -------------------------------------------------
    def observed_lines(key: str) -> list[str]:
        im = ext_images[key]
        return [
            fmt(a["cls"], coco_to_yolo(a["bbox"], im["width"], im["height"],
                                       f"observed box on {key}"))
            for a in ext_observed[key]
        ]

    a00 = {k: observed_lines(k) for k in wanted}
    a10 = {k: a00[k] + additions[ACTIVE][k] for k in wanted}
    a11 = {k: a10[k] + additions[NON_ACTIVE][k] for k in wanted}

    val_images = sorted(
        p for p in (args.eyecu_root / "images" / "val").iterdir() if p.is_file()
    )

    reports = []
    for arm, labels in (("A00", a00), ("A10", a10), ("A11", a11)):
        reports.append(build_arm(arm, out, args.eyecu_root, args.source_images,
                                 ext_images, labels, val_images))

    # -- hard assertions ----------------------------------------------------
    by_arm = {r["arm"]: r for r in reports}
    for r in reports:
        if r["train_images"] != expected["total"]:
            fail(f"{r['arm']} has {r['train_images']} train images, "
                 f"expected {expected['total']}")
        if r["eyecu_train_images"] != expected["eyecu_train"]:
            fail(f"{r['arm']} EyeCU count is {r['eyecu_train_images']}")
        if r["external_train_images"] != expected["external_train"]:
            fail(f"{r['arm']} external count is {r['external_train_images']}")

    stems = {r["train_stems_sha256"] for r in reports}
    if len(stems) != 1:
        fail("the three arms do not hold identical training image ids")

    d_a10 = by_arm["A10"]["ball_labels"] - by_arm["A00"]["ball_labels"]
    d_a11 = by_arm["A11"]["ball_labels"] - by_arm["A10"]["ball_labels"]
    if d_a10 != expected["active"]:
        fail(f"A00->A10 added {d_a10} ball labels, expected {expected['active']}")
    if d_a11 != expected["nonactive"]:
        fail(f"A10->A11 added {d_a11} ball labels, expected {expected['nonactive']}")
    for arm in ("A10", "A11"):
        for cls in ("player", "goalkeeper", "referee"):
            if by_arm[arm]["labels_by_class"].get(cls) != \
                    by_arm["A00"]["labels_by_class"].get(cls):
                fail(f"{arm} changed the {cls} class; Stage-A only adds footballs")

    manifest = {
        "built_by": "tools/build_stage_a_dataset.py",
        "design": str(args.design),
        "ontology": "ALL_VISIBLE_PHYSICAL_FOOTBALLS",
        "class_schema": EYECU_NAMES,
        "class_remap_by_name": SOURCE_TO_EYECU_NAME,
        "expected": expected,
        "round0_split_census": dict(census),
        "source_export_sha256": sha256_file(args.source_export),
        "eyecu_train_labels_sha256": sha256_tree(
            list((args.eyecu_root / "labels" / "train").glob("*.txt"))),
        "arms": reports,
        "additions": {
            "ACTIVE": role_census[ACTIVE],
            "NON_ACTIVE": role_census[NON_ACTIVE],
            "total": sum(role_census.values()),
        },
        "leakage": {
            "source_valid_in_train": leak_valid,
            "source_test_in_train": leak_test,
            "eyecu_sealed_test_exposure": 0,
        },
        "image_ids_identical_across_arms": True,
    }
    (out / "STAGE_A_BUILD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")

    hdr = f"{'':<26}{'A00':>12}{'A10':>12}{'A11':>12}"
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("train images", "train_images"),
        ("  EyeCU", "eyecu_train_images"),
        ("  external source-TRAIN", "external_train_images"),
        ("val images", "val_images"),
        ("total labels", "total_labels"),
        ("ball labels", "ball_labels"),
    ]
    for label, key in rows:
        print(f"{label:<26}" + "".join(f"{by_arm[a][key]:>12}" for a in
                                       ("A00", "A10", "A11")))
    print(f"{'ACTIVE additions':<26}{0:>12}{d_a10:>12}{d_a10:>12}")
    print(f"{'NON_ACTIVE additions':<26}{0:>12}{0:>12}{d_a11:>12}")
    print()
    print(f"image ids identical across arms : YES ({stems.pop()[:16]}...)")
    print(f"source-valid leakage            : {leak_valid}")
    print(f"source-test leakage             : {leak_test}")
    print("eyecu sealed-test exposure      : 0")
    print(f"\nmanifest: {out / 'STAGE_A_BUILD_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BuildError as exc:
        print(f"STAGE-A BUILD FAILED: {exc}", file=sys.stderr)
        sys.exit(2)
